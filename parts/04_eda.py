# %% [markdown]
# ## 7. Análise exploratória
# Imagens originais, máscaras, *overlays* e distribuição de classes. Os dois **frames** de avaliação/treino são:
# * `padded` — tela 256×256 do artigo (conteúdo no canto, resto preto);
# * `roi` — apenas o conteúdo original, depois ampliado (*letterbox*) para a tela do modelo. A avaliação reverte para o tamanho original.

# %%
def load_frame_records(frame: str) -> dict[str, list[dict]]:
    """Carrega imagens+máscaras anotadas num dado frame: {'train': [...], 'val': [...], 'test': [...]}."""
    assert frame in ("roi", "padded")
    folder = "Original" if frame == "roi" else "Padded"
    out: dict[str, list[dict]] = {}
    for split, names in SPLIT_NAMES.items():
        recs = []
        for n in names:
            img = read_rgb(_locate(folder, "Images", n))
            mpath = _locate(folder, "Palette", n) or _locate(folder, "Annotations", n)
            mask, _ = read_mask(mpath)
            assert img.shape[:2] == mask.shape, f"{n}: imagem {img.shape[:2]} != máscara {mask.shape}"
            recs.append(dict(name=n, img=img, mask=mask))
        out[split] = recs
    return out


def load_unlabeled_records(frame: str) -> list[dict]:
    recs = []
    for r in UNL:
        img = read_rgb(r["path"])
        if frame == "roi":
            img = np.ascontiguousarray(img[: r["h"], : r["w"]])
        recs.append(dict(name=r["name"], img=img))
    return recs


_FRAME_CACHE: dict[str, dict] = {}


def get_frame(frame: str) -> dict[str, list[dict]]:
    if frame not in _FRAME_CACHE:
        _FRAME_CACHE[frame] = load_frame_records(frame)
    return _FRAME_CACHE[frame]


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32).copy()
    fg = mask > 0
    out[fg] = (1 - alpha) * out[fg] + alpha * PALETTE[mask][fg]
    return out.astype(np.uint8)


def savefig(fig, name: str) -> None:
    fig.savefig(DIRS["figures"] / name, dpi=130, bbox_inches="tight")


_orig = get_frame("roi")
fig, axes = plt.subplots(3, 6, figsize=(16, 8.5))
for j, rec in enumerate(_orig["train"][:6]):
    axes[0, j].imshow(rec["img"]); axes[0, j].set_title(f"{rec['name']}  {rec['img'].shape[1]}x{rec['img'].shape[0]}", fontsize=9)
    axes[1, j].imshow(colorize(rec["mask"]))
    axes[2, j].imshow(overlay(rec["img"], rec["mask"]))
    for i in range(3):
        axes[i, j].axis("off")
axes[0, 0].set_ylabel("imagem"); fig.suptitle("Treino — imagem / máscara (vermelho=fibrina, verde=granulação, azul=calo) / overlay (frame original)")
fig.tight_layout(); savefig(fig, "eda_samples.png"); plt.show()

# %%
# Distribuição de classes, tamanhos e o efeito do padding
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
share.T.plot.bar(ax=ax[0], logy=True, rot=20, legend=True); ax[0].set_title("Píxeis por classe e divisão (log)")
presence.T.plot.bar(ax=ax[1], rot=20); ax[1].set_title("Nº de imagens com a classe")
ax[2].hist(AUDIT.h, bins=20); ax[2].set_title("Altura original (px)"); ax[2].axvline(256, color="r", ls="--")
_content_frac = (AUDIT.h * AUDIT.w).clip(upper=256 * 256) / (256 * 256)
ax[3].hist(_content_frac, bins=20); ax[3].set_title("Fração da tela 256x256 ocupada pelo conteúdo\n(no frame 'padded')")
fig.tight_layout(); savefig(fig, "eda_distribution.png"); plt.show()
print(f"Mediana da fração útil da tela no frame 'padded': {_content_frac.median():.2f} (o resto é preto)")

if UNL:
    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    for a, r in zip(axes, UNL[:6]):
        a.imshow(read_rgb(r["path"])[: r["h"], : r["w"]]); a.set_title(f"não anotada {r['name']}", fontsize=8); a.axis("off")
    fig.tight_layout(); savefig(fig, "eda_unlabeled.png"); plt.show()
