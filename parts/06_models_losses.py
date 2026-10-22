# %% [markdown]
# ## 10. Arquiteturas e pesos pré-treinados
#
# Três configurações complementares (relação desempenho/estabilidade/custo), todas com normalização ImageNet embutida no modelo:
#
# | id | Arquitetura | Encoder (pesos) | Papel |
# |---|---|---|---|
# | `unetpp_effb4` | U-Net++ (`smp`) | EfficientNet-B4 `ra2_in1k` (timm) | CNN eficiente, *skip connections* densas — **referência** |
# | `upernet_convnext_s` | UPerNet (`smp`) | ConvNeXt-S `fb_in22k_ft_in1k` (timm) | CNN moderna com pirâmide multiescala/contexto global |
# | `segformer_mitb3` | SegFormer (`smp`) | MiT-B3 ImageNet-1k (NVlabs, via `smp`) | Transformer hierárquico (encoder do artigo, mas **sem** P-scSE) |
#
# **Origem dos pesos.** `build_model` tenta, por esta ordem: (1) descarregar pesos pré-treinados (precisa de Internet); (2) carregar um
# ficheiro local (`.pth/.pt/.bin/.safetensors`) encontrado em `/kaggle/input` cujo nome/pasta contenha a chave do encoder
# (p.ex. `efficientnet_b4`, `convnext_small`, `mit_b3`); (3) **pesos aleatórios, com aviso e registo**. A origem de cada execução é
# gravada em `weights_provenance.json` (inclui `source` e a nota de licença **a confirmar no *model card***). Os pesos MiT do
# SegFormer têm licença da NVIDIA (uso não comercial); confirme as licenças dos restantes antes de reutilizar.

# %%
ARCH_SPECS = {
    "unetpp_effb4": dict(
        ctor=lambda w: smp.UnetPlusPlus("tu-efficientnet_b4.ra2_in1k", encoder_weights=w, classes=K, in_channels=3),
        key="efficientnet_b4", family="CNN", origin="timm efficientnet_b4.ra2_in1k (ImageNet-1k, RandAugment recipe)",
        license="ver model card timm/efficientnet_b4.ra2_in1k (Apache-2.0 declarado no código timm; confirmar nos pesos)"),
    "upernet_convnext_s": dict(
        ctor=lambda w: smp.UPerNet("tu-convnext_small.fb_in22k_ft_in1k", encoder_weights=w, classes=K, in_channels=3),
        key="convnext_small", family="CNN", origin="timm convnext_small.fb_in22k_ft_in1k (ImageNet-22k -> 1k, Meta AI)",
        license="ver model card timm/convnext_small.fb_in22k_ft_in1k (código ConvNeXt: MIT; confirmar nos pesos)"),
    "segformer_mitb3": dict(
        ctor=lambda w: smp.Segformer("mit_b3", encoder_weights=w, classes=K, in_channels=3),
        key="mit_b3", family="Transformer", origin="NVlabs SegFormer MiT-B3 ImageNet-1k (distribuído por smp / HF)",
        license="NVIDIA Source Code License (não comercial) — confirmar antes de uso comercial"),
    "smoke_fpn_r18": dict(  # apenas para o modo smoke (teste de pipeline)
        ctor=lambda w: smp.FPN("resnet18", encoder_weights=w, classes=K, in_channels=3),
        key="resnet18", family="CNN", origin="torchvision resnet18 (ImageNet-1k)", license="BSD-3 (torchvision)"),
}
WEIGHTS_PROVENANCE: dict[str, dict] = {}
_WARNED_RANDOM: set[str] = set()


class Net(nn.Module):
    """Entrada em [0,1]; normalização ImageNet interna; saída: *logits* (B,K,H,W)."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core((x - self.mean) / self.std)

    def encoder_params(self):
        return list(self.core.encoder.parameters())

    def other_params(self):
        enc = {id(p) for p in self.core.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc]


def _local_weight_files(key: str, max_depth: int = 5) -> list[Path]:
    out = []
    if INPUT_ROOT.exists():
        d0 = len(INPUT_ROOT.parts)
        for dp, dn, fn in os.walk(INPUT_ROOT):
            if len(Path(dp).parts) - d0 >= max_depth:
                dn[:] = []
            for f in fn:
                if f.lower().endswith((".pth", ".pt", ".bin", ".safetensors")) and key in (dp + "/" + f).lower():
                    out.append(Path(dp) / f)
    return out


def _read_state(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("state_dict", "model", "state"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _match_ratio(sd: dict, target: dict) -> tuple[dict, float]:
    """Casa chaves (também com 'a.b.c' -> 'a_b.c', como no `features_only` do timm) e devolve fração dos parâmetros-alvo cobertos."""
    matched = {}
    for k, v in sd.items():
        cands = [k] + [k.replace(".", "_", n) for n in (1, 2, 3)]
        for c in cands:
            if c in target and target[c].shape == v.shape:
                matched[c] = v
                break
    tot = sum(t.numel() for t in target.values())
    cov = sum(t.numel() for k, t in target.items() if k in matched)
    return matched, cov / max(tot, 1)


def load_local_encoder(core: nn.Module, path: Path) -> float:
    sd = _read_state(path)
    best = (None, None, 0.0)
    for tgt in [getattr(core.encoder, "model", None), core.encoder]:
        if tgt is None:
            continue
        m, r = _match_ratio(sd, tgt.state_dict())
        if r > best[2]:
            best = (tgt, m, r)
    if best[0] is not None and best[2] >= 0.8:
        best[0].load_state_dict(best[1], strict=False)
    return best[2]


def build_model(arch: str, pretrained: bool = True) -> tuple[Net, dict]:
    spec = ARCH_SPECS[arch]
    prov = dict(arch=arch, family=spec["family"], origin=spec["origin"], license_note=spec["license"], requested_pretrained=pretrained, source="random_init")
    core = None
    if pretrained:
        try:
            core = spec["ctor"]("imagenet")
            prov["source"] = "downloaded_or_cached_pretrained"
        except Exception as e:  # noqa: BLE001 — sem Internet / sem cache
            prov["online_error"] = f"{type(e).__name__}: {str(e)[:160]}"
            files = _local_weight_files(spec["key"])
            for f in files:
                core = spec["ctor"](None)
                ratio = load_local_encoder(core, f)
                if ratio >= 0.8:
                    prov.update(source=f"local_file:{f}", local_match_ratio=round(ratio, 3))
                    break
                core = None
            if core is None and arch not in _WARNED_RANDOM:
                _WARNED_RANDOM.add(arch)
                print(f"AVISO [{arch}]: pesos pré-treinados indisponíveis (Internet off e sem ficheiro local). A usar INICIALIZAÇÃO ALEATÓRIA — "
                      f"o desempenho será pior e a comparação não é válida como 'transfer learning'.")
    if core is None:
        core = spec["ctor"](None)
        prov["pretrained_effective"] = False
    else:
        prov["pretrained_effective"] = True
    WEIGHTS_PROVENANCE[arch] = prov
    return Net(core), prov


def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


# %% [markdown]
# ## 11. Funções de perda
# `ce_dice` (padrão): *cross-entropy* ponderada por classe (pesos ∝ 1/√frequência, **limitados** a `class_weight_cap` para não inflacionar falsos positivos)
# + Dice multiclasse sobre os 3 tecidos. Alternativas comparadas na validação em modo `full`: `focal_dice`, `focal_tversky`
# (Tversky α=0,3, β=0,7, γ=0,75 + 0,5·CE) e `ce_dice_lovasz`. `ignore_index=255` suporta *pseudo-labels* incertos no SSL.

# %%
IGNORE = 255


def class_weights_from(train_recs: list[dict], cap: float) -> torch.Tensor:
    cnt = np.zeros(K)
    for r in train_recs:
        cnt += np.bincount(r["mask"].ravel(), minlength=K)[:K]
    freq = cnt / cnt.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    w = np.minimum(w / w.min(), cap)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class SegLoss(nn.Module):
    def __init__(self, name: str, class_w: torch.Tensor | None, label_smoothing: float = 0.0):
        super().__init__()
        self.name = name
        self.register_buffer("cw", class_w if class_w is not None else torch.ones(K))
        self.ls = label_smoothing
        self.dice = smp.losses.DiceLoss("multiclass", classes=TISSUES, from_logits=True, ignore_index=IGNORE)
        self.focal = smp.losses.FocalLoss("multiclass", gamma=2.0, ignore_index=IGNORE)
        self.tversky = smp.losses.TverskyLoss("multiclass", classes=TISSUES, from_logits=True, alpha=0.3, beta=0.7, gamma=0.75, ignore_index=IGNORE)
        self.lovasz = smp.losses.LovaszLoss("multiclass", per_image=False, from_logits=True, ignore_index=IGNORE)
        assert name in ("ce", "ce_dice", "focal_dice", "focal_tversky", "ce_dice_lovasz"), name

    def ce(self, logits, y):
        return F.cross_entropy(logits, y, weight=self.cw, ignore_index=IGNORE, label_smoothing=self.ls)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        n = self.name
        if n == "ce":
            return self.ce(logits, y)
        if n == "ce_dice":
            return self.ce(logits, y) + self.dice(logits, y)
        if n == "focal_dice":
            return self.focal(logits, y) + self.dice(logits, y)
        if n == "focal_tversky":
            return self.tversky(logits, y) + 0.5 * self.ce(logits, y)
        return self.ce(logits, y) + self.dice(logits, y) + 0.5 * self.lovasz(logits, y)


def _loss_selftest():
    torch.manual_seed(0)
    lg = torch.randn(2, K, 32, 32, requires_grad=True)
    y = torch.randint(0, K, (2, 32, 32))
    y[0, :8] = IGNORE
    cw = torch.tensor([1.0, 3.0, 2.0, 1.5])
    for n in ["ce", "ce_dice", "focal_dice", "focal_tversky", "ce_dice_lovasz"]:
        l = SegLoss(n, cw)(lg, y)
        l.backward()
        assert torch.isfinite(l) and lg.grad is not None and torch.isfinite(lg.grad).all(), n
        lg.grad = None
    # perda ~0 quando o logit aponta com muita confiança para a classe certa
    y2 = torch.randint(0, K, (2, 32, 32))
    perfect = F.one_hot(y2, K).permute(0, 3, 1, 2).float() * 30
    assert SegLoss("ce_dice", None)(perfect, y2) < 0.05
    print("OK: testes das funções de perda (finitas, gradiente, ignore_index, ~0 no caso perfeito)")


_loss_selftest()
