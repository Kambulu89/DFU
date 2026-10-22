# %% [markdown]
# ## 8–9. Pré-processamento, *augmentation*, *Datasets* e *DataLoaders*
#
# * **Letterbox**: o conteúdo (altura×largura originais) é ampliado/reduzido mantendo a proporção até caber numa tela `S×S` e centrado.
#   Imagens e máscaras partilham exatamente a mesma geometria; **máscaras usam sempre *nearest***, probabilidades usam *bilinear* só ao reverter.
# * **Augmentation** própria (OpenCV/NumPy, sem dependência de versão de bibliotecas): flips, afim moderada (rotação ±25°, escala 0,8–1,25,
#   translação ±8 %, cisalhamento ±5°), HSV suave (matiz ±5 em escala 0–179 de OpenCV), brilho/contraste, gamma, ruído e desfoque.
#   A fotometria só altera a **área válida** (o preto do *padding* mantém-se preto, como nas imagens de teste). Não se usam *elastic/grid distortion*
#   (alteram fronteiras já ambíguas) nem *copy-paste* (criaria tecidos anatomicamente incoerentes). Se uma transformação geométrica destruir
#   >60 % do primeiro plano, repete-se (preserva a região de interesse). **Nada aleatório é aplicado na avaliação.**
# * **Class-aware sampling**: imagens com mais fibrina têm maior probabilidade de amostragem (`fibrin_boost`).

# %%
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_geom(h: int, w: int, S: int) -> dict:
    s = S / max(h, w)
    nh, nw = max(1, min(S, round(h * s))), max(1, min(S, round(w * s)))
    return dict(h=h, w=w, S=S, scale=s, nh=nh, nw=nw, y0=(S - nh) // 2, x0=(S - nw) // 2)


def apply_letterbox(img: np.ndarray, mask: np.ndarray | None, g: dict):
    """Devolve (tela imagem uint8 SxSx3, tela máscara uint8 SxS | None, máscara de validade uint8 SxS)."""
    S, nh, nw, y0, x0 = g["S"], g["nh"], g["nw"], g["y0"], g["x0"]
    interp = cv2.INTER_AREA if g["scale"] < 1 else cv2.INTER_CUBIC
    canvas = np.zeros((S, S, 3), np.uint8)
    canvas[y0:y0 + nh, x0:x0 + nw] = np.clip(cv2.resize(img, (nw, nh), interpolation=interp), 0, 255)
    cm = None
    if mask is not None:
        cm = np.zeros((S, S), np.uint8)
        cm[y0:y0 + nh, x0:x0 + nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)  # NEAREST: rótulos categóricos
    valid = np.zeros((S, S), np.uint8)
    valid[y0:y0 + nh, x0:x0 + nw] = 1
    return canvas, cm, valid


def unletterbox_probs(probs: torch.Tensor, g: dict) -> torch.Tensor:
    """(C,S,S) -> (C,h,w): recorta a área do conteúdo e repõe o tamanho original (bilinear nas probabilidades)."""
    crop = probs[:, g["y0"]:g["y0"] + g["nh"], g["x0"]:g["x0"] + g["nw"]]
    if (g["nh"], g["nw"]) == (g["h"], g["w"]):
        return crop
    return F.interpolate(crop[None].float(), size=(g["h"], g["w"]), mode="bilinear", align_corners=False)[0]


AUG_LEVELS = {
    "none": dict(),
    "flip": dict(flip_h=0.5, flip_v=0.5),
    "weak": dict(flip_h=0.5, flip_v=0.5, affine_p=0.5, rot=10, scale=(0.9, 1.1), trans=0.04, shear=0),
    "standard": dict(flip_h=0.5, flip_v=0.5, affine_p=0.8, rot=25, scale=(0.8, 1.25), trans=0.08, shear=5,
                     hsv_p=0.4, bc_p=0.6, gamma_p=0.3, noise_p=0.25, blur_p=0.2),
}


class Augmenter:
    """Augmentation geométrica+fotométrica; imagem e máscara são alinhadas pela mesma matriz afim."""

    def __init__(self, level: str = "standard"):
        self.p = AUG_LEVELS[level]
        self.level = level

    def __call__(self, img: np.ndarray, mask: np.ndarray | None, valid: np.ndarray, rng: np.random.Generator):
        p, S = self.p, img.shape[0]
        if not p:
            return img, mask, valid
        if rng.random() < p.get("flip_h", 0):
            img, valid = img[:, ::-1], valid[:, ::-1]
            mask = mask[:, ::-1] if mask is not None else None
        if rng.random() < p.get("flip_v", 0):
            img, valid = img[::-1], valid[::-1]
            mask = mask[::-1] if mask is not None else None
        img, valid = np.ascontiguousarray(img), np.ascontiguousarray(valid)
        mask = np.ascontiguousarray(mask) if mask is not None else None
        if rng.random() < p.get("affine_p", 0):
            fg0 = int((mask > 0).sum()) if mask is not None else 0
            for _ in range(4):
                ang = rng.uniform(-p["rot"], p["rot"])
                sc = math.exp(rng.uniform(math.log(p["scale"][0]), math.log(p["scale"][1])))
                sh = math.radians(rng.uniform(-p["shear"], p["shear"])) if p["shear"] else 0.0
                tx, ty = rng.uniform(-p["trans"], p["trans"], 2) * S
                M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, sc)
                M = np.vstack([M, [0, 0, 1]]) @ np.array([[1, math.tan(sh), -math.tan(sh) * S / 2], [0, 1, 0], [0, 0, 1]])
                M[0, 2] += tx
                M[1, 2] += ty
                M = M[:2]
                m2 = cv2.warpAffine(mask, M, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) if mask is not None else None
                if m2 is None or fg0 == 0 or (m2 > 0).sum() >= 0.4 * fg0:
                    break
            img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            valid = cv2.warpAffine(valid, M, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = m2
        if "hsv_p" in p or "bc_p" in p:
            img = self._photometric(img, valid, rng)
        return img, mask, valid

    def _photometric(self, img: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = self.p
        if rng.random() < p["hsv_p"]:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.9, 1.1), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        x = img.astype(np.float32) / 255.0
        if rng.random() < p["bc_p"]:
            x = (x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1)
        if rng.random() < p["gamma_p"]:
            x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)
        if rng.random() < p["blur_p"]:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.0))
        if rng.random() < p["noise_p"]:
            x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape).astype(np.float32)
        x = np.clip(x, 0, 1) * valid[..., None]  # o padding mantém-se exatamente preto
        return (x * 255 + 0.5).astype(np.uint8)


def to_tensor01(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1).float().div_(255.0)


class SegSet(torch.utils.data.Dataset):
    """Dataset de treino (anotado ou não). Devolve tensores em [0,1]; a normalização ImageNet faz parte do modelo."""

    def __init__(self, recs: list[dict], S: int, aug: str = "standard", labeled: bool = True):
        self.items = []
        for r in recs:
            g = letterbox_geom(*r["img"].shape[:2], S)
            c, m, v = apply_letterbox(r["img"], r["mask"] if labeled else None, g)
            self.items.append(dict(name=r["name"], img=c, mask=m, valid=v))
        self.aug, self.labeled, self._rng = Augmenter(aug), labeled, None

    def __len__(self) -> int:
        return len(self.items)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng((torch.initial_seed() + os.getpid()) % (2**32))
        return self._rng

    def __getitem__(self, i: int):
        it = self.items[i]
        img, mask, _ = self.aug(it["img"], it["mask"], it["valid"], self._get_rng())
        x = to_tensor01(img)
        if self.labeled:
            return x, torch.from_numpy(np.ascontiguousarray(mask)).long()
        return x

    def sample_weights(self, boost: float) -> np.ndarray:
        """w_i = 1 + boost * (fração de fibrina no 1º plano / média), limitado a [0, 2]. Só usa máscaras de TREINO."""
        fr = np.array([((it["mask"] == 1).sum() / max(1, (it["mask"] > 0).sum())) for it in self.items])
        return 1.0 + boost * np.clip(fr / max(fr.mean(), 1e-6), 0, 2.0)


class EvalItems:
    """Itens de avaliação: tela de entrada (sem augmentation) + geometria + GT no frame de avaliação (tamanho do conteúdo)."""

    def __init__(self, recs: list[dict], S: int, labeled: bool = True):
        self.names, self.geoms, self.gts, self.imgs, canv, cgt = [], [], [], [], [], []
        for r in recs:
            g = letterbox_geom(*r["img"].shape[:2], S)
            c, cm, _ = apply_letterbox(r["img"], r["mask"] if labeled else None, g)
            self.names.append(r["name"]); self.geoms.append(g); self.imgs.append(r["img"])
            self.gts.append(r["mask"] if labeled else None)
            canv.append(to_tensor01(c))
            if labeled:
                cgt.append(torch.from_numpy(cm).long())
        self.canvas = torch.stack(canv)
        self.canvas_gt = torch.stack(cgt) if labeled else None  # GT na tela do modelo (só para a perda de validação)

    def __len__(self) -> int:
        return len(self.names)


def worker_init(_):  # cada worker cria o seu gerador na primeira utilização (semente derivada de torch.initial_seed)
    cv2.setNumThreads(0)


def make_train_loader(train_set: SegSet, batch: int, boost: float, seed: int) -> torch.utils.data.DataLoader:
    w = torch.as_tensor(train_set.sample_weights(boost), dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train_set), replacement=True, generator=g)
    nw = CFG["num_workers"]
    return torch.utils.data.DataLoader(train_set, batch_size=batch, sampler=sampler, num_workers=nw, drop_last=len(train_set) > batch,
                                       pin_memory=DEVICE.type == "cuda", persistent_workers=nw > 0, worker_init_fn=worker_init, generator=g)


# %%
# Verificação visual do alinhamento imagem/máscara sob augmentation (as máscaras nunca são interpoladas)
_fr = get_frame(CFG["frame"])
_demo = SegSet(_fr["train"][:4], CFG["size"], aug="standard")
_demo._rng = np.random.default_rng(0)
fig, axes = plt.subplots(3, 6, figsize=(16, 8.3))
for j in range(6):
    x, m = _demo[j % 4]
    im = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    axes[0, j].imshow(im); axes[1, j].imshow(colorize(m.numpy().astype(np.uint8))); axes[2, j].imshow(overlay(im, m.numpy().astype(np.uint8)))
    axes[0, j].set_title("augmentada", fontsize=8)
    for i in range(3):
        axes[i, j].axis("off")
fig.suptitle(f"Augmentation de treino (frame={CFG['frame']}, tela={CFG['size']}) — imagem, máscara, overlay")
fig.tight_layout(); savefig(fig, "aug_examples.png"); plt.show()
# teste automático: valores da máscara continuam em {0,1,2,3} após augmentation (sem interpolação de rótulos)
for _ in range(50):
    _x, _m = _demo[random.randrange(4)]
    assert set(torch.unique(_m).tolist()) <= {0, 1, 2, 3}
print("OK: máscaras augmentadas só contêm rótulos 0..3")
