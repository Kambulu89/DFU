# %% [markdown]
# ## 6. Localização e auditoria do dataset
#
# O notebook procura em `/kaggle/input` (ou em `DFU_DATA_ROOT`) uma pasta que contenha `Labeled/labeled_train_names.txt`
# (estrutura do repositório oficial: `Labeled/{Original,Padded}/{Images,Annotations,Palette}/{TrainVal,Test}`, `Labeled/*_names.txt`
# e `Unlabeled/`). **Não inventa dados**: se não os encontrar, para com instruções.
#
# Verificações: contagens, correspondência imagem↔máscara, ficheiros corrompidos, valores únicos e codificação das máscaras
# (indexadas `P`, cinzentos `L` ou RGB — **não se assume** um formato), tamanhos, equivalência *Padded* ≙ *Original*,
# distribuição de píxeis, duplicados exatos e quase-duplicados (*dHash*) dentro e **entre** divisões, e entre os 600 não anotados e val/teste.

# %%
ALLOW_GIT_DOWNLOAD = os.environ.get("DFU_ALLOW_DOWNLOAD", "0") == "1"  # opt-in; só com Internet ligada
DATA_HELP = """
Dataset DFUTissue não encontrado. Passos para o adicionar ao Kaggle:
  1. Obtenha a pasta 'DFUTissue' de https://github.com/uwm-bigdata/DFUTissueSegNet (contém Labeled/ e Unlabeled/).
  2. Kaggle > Datasets > New Dataset > carregue a pasta (mantenha a estrutura Labeled/ e Unlabeled/).
  3. No notebook: Add Input > o seu dataset. Em alternativa defina DFU_DATA_ROOT ou DFU_ALLOW_DOWNLOAD=1 (requer Internet).
"""


def find_dataset_root() -> Path | None:
    explicit = os.environ.get("DFU_DATA_ROOT")
    bases = [Path(explicit)] if explicit else [INPUT_ROOT]
    for base in bases:
        if not base.exists():
            continue
        depth0 = len(base.parts)
        for dirpath, dirnames, filenames in os.walk(base):
            if len(Path(dirpath).parts) - depth0 > 6:
                dirnames[:] = []
                continue
            if "labeled_train_names.txt" in filenames and Path(dirpath).name == "Labeled":
                return Path(dirpath).parent
    return None


DATA_ROOT = find_dataset_root()
if DATA_ROOT is None and ALLOW_GIT_DOWNLOAD and INTERNET:
    dst = WORK / "DFUTissueSegNet"
    if not dst.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/uwm-bigdata/DFUTissueSegNet.git", str(dst)], check=True)
    DATA_ROOT = dst / "DFUTissue"
if DATA_ROOT is None:
    raise RuntimeError(DATA_HELP)
LAB_DIR = DATA_ROOT / "Labeled"
UNL_DIR = DATA_ROOT / "Unlabeled" if (DATA_ROOT / "Unlabeled").exists() else None
print("DATA_ROOT:", DATA_ROOT, "| Unlabeled:", "sim" if UNL_DIR else "NÃO (SSL será omitido)")


def read_names(fname: str) -> list[str]:
    return [l.strip() for l in (LAB_DIR / fname).read_text().splitlines() if l.strip()]


SPLIT_NAMES = {"train": read_names("labeled_train_names.txt"), "val": read_names("labeled_val_names.txt"), "test": read_names("test_names.txt")}
print({k: len(v) for k, v in SPLIT_NAMES.items()}, "(artigo: 78 / 16 / 16)")


def _locate(frame_dir: str, kind: str, name: str) -> Path | None:
    for sub in ("TrainVal", "Test", "Train", "Val", ""):
        p = LAB_DIR / frame_dir / kind / sub / f"{name}.png"
        if p.exists():
            return p
    return None


MASK_COLORS = {(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2, (0, 0, 255): 3}


def read_mask(path: Path) -> tuple[np.ndarray, str]:
    """Lê uma máscara e devolve (índices uint8 [0..3], descrição da codificação). Falha se houver valores inválidos."""
    im = Image.open(path)
    mode, arr = im.mode, np.array(im)
    if mode in ("P", "L", "I", "I;16"):
        idx, enc = arr.astype(np.int64), f"{mode}:indices"
    elif mode in ("RGB", "RGBA"):
        rgb = arr[..., :3]
        if (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all():
            idx, enc = rgb[..., 0].astype(np.int64), f"{mode}:canais-iguais(indices)"
        else:
            idx = np.full(rgb.shape[:2], -1, np.int64)
            for col, k in MASK_COLORS.items():
                idx[(rgb == np.array(col)).all(-1)] = k
            enc = f"{mode}:cores"
    else:
        raise ValueError(f"Modo de máscara não suportado: {mode} ({path})")
    if idx.min() < 0 or idx.max() > 3:
        raise ValueError(f"Máscara com valores fora de 0..3 ou cores desconhecidas: {path} -> {np.unique(idx)[:10]}")
    return idx.astype(np.uint8), enc


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def content_extent(img: np.ndarray) -> tuple[int, int]:
    """Altura/largura do conteúdo numa imagem com *zero padding* no canto inferior-direito."""
    nz = img.max(-1) > 0
    rows, cols = np.where(nz.any(1))[0], np.where(nz.any(0))[0]
    return int(rows.max()) + 1, int(cols.max()) + 1


def dhash(img: np.ndarray) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# %%
audit_rows, problems = [], []
labeled_hash: dict[str, int] = {}
for split, names in SPLIT_NAMES.items():
    for n in names:
        row = dict(name=n, split=split, corrupt=False)
        try:
            pi, pa, pp = _locate("Original", "Images", n), _locate("Original", "Annotations", n), _locate("Original", "Palette", n)
            qi, qa, qp = _locate("Padded", "Images", n), _locate("Padded", "Annotations", n), _locate("Padded", "Palette", n)
            if pi is None or pa is None or qi is None or (qa is None and qp is None):
                raise FileNotFoundError("imagem/máscara em falta")
            img = read_rgb(pi)
            ann, enc_a = read_mask(pa)
            row.update(h=img.shape[0], w=img.shape[1], enc_orig_annotation=enc_a, size_mismatch=img.shape[:2] != ann.shape)
            if pp is not None:
                pal, enc_p = read_mask(pp)
                row.update(enc_orig_palette=enc_p, annot_eq_palette=bool((pal == ann).all()))
            pimg = read_rgb(qi)
            pm, enc_pm = read_mask(qa or qp)
            row.update(enc_padded_mask=enc_pm, padded_shape=pimg.shape[:2])
            h, w = img.shape[:2]
            if max(h, w) <= pimg.shape[0]:
                row["padded_img_eq_orig"] = bool((pimg[:h, :w] == img).all() and (pimg[h:] == 0).all() and (pimg[:, w:] == 0).all())
                row["padded_mask_eq_orig"] = bool((pm[:h, :w] == ann).all() and (pm[h:] == 0).all() and (pm[:, w:] == 0).all())
                row["extent_detected_ok"] = content_extent(pimg) == (h, w)
            else:  # imagens > 256 foram reduzidas no 'Padded' (documentado no relatório)
                row["padded_img_eq_orig"] = row["padded_mask_eq_orig"] = row["extent_detected_ok"] = None
            counts = np.bincount(ann.ravel(), minlength=4)[:4]
            row.update({f"px_{c.lower()}": int(v) for c, v in zip(CLASS_NAMES, counts)})
            labeled_hash[n] = dhash(img)
        except Exception as e:  # noqa: BLE001 — a auditoria deve reportar, não rebentar
            row.update(corrupt=True, error=f"{type(e).__name__}: {e}")
            problems.append((n, row["error"]))
        audit_rows.append(row)

AUDIT = pd.DataFrame(audit_rows)
AUDIT.to_csv(DIRS["tables"] / "audit_labeled.csv", index=False)
print(f"Imagens anotadas: {len(AUDIT)} | corrompidas/inválidas: {int(AUDIT.corrupt.sum())}")
for n, e in problems:
    print("  PROBLEMA", n, e)
assert not AUDIT.corrupt.any(), "Há ficheiros inválidos — corrija o dataset antes de treinar."
print("Codificações (Original/Annotations):", AUDIT.enc_orig_annotation.value_counts().to_dict())
print("Codificações (Padded mask):", AUDIT.enc_padded_mask.value_counts().to_dict())
if "annot_eq_palette" in AUDIT:
    print("Annotations == Palette em todas:", bool(AUDIT.annot_eq_palette.dropna().all()))
print("Imagem/máscara com tamanhos diferentes:", int(AUDIT.size_mismatch.sum()))
print("Padded == Original (conteúdo no canto sup-esq.):", AUDIT.padded_img_eq_orig.dropna().all(), "| máscaras:", AUDIT.padded_mask_eq_orig.dropna().all(),
      "| deteção de extensão correta:", AUDIT.extent_detected_ok.dropna().all(), f"| imagens >256 (não comparáveis): {int(AUDIT.padded_img_eq_orig.isna().sum())}")
print("Tamanhos originais: min", AUDIT[["h", "w"]].min().tolist(), "max", AUDIT[["h", "w"]].max().tolist(), "| mediana", AUDIT[["h", "w"]].median().tolist())

# divisões: sobreposição por nome
sets = {k: set(v) for k, v in SPLIT_NAMES.items()}
overlaps = {f"{a}&{b}": sorted(sets[a] & sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
print("Sobreposição de nomes entre divisões:", overlaps)
assert all(len(v) == 0 for v in overlaps.values()), "Fuga: nomes repetidos entre divisões"

# %%
# Distribuição de píxeis por classe e por divisão (frame original)
px_cols = [f"px_{c.lower()}" for c in CLASS_NAMES]
share = AUDIT.groupby("split")[px_cols].sum()
share_pct = share.div(share.sum(axis=1), axis=0) * 100
presence = AUDIT.groupby("split")[px_cols].agg(lambda s: int((s > 0).sum()))
fg_share = share[px_cols[1:]].div(share[px_cols[1:]].sum(axis=1), axis=0) * 100
print("% de píxeis por classe (inclui fundo):")
display(share_pct.round(2))
print("% de píxeis do primeiro plano:")
display(fg_share.round(1))
print("Nº de imagens com cada classe:")
display(presence)

# Duplicados exatos e quase-duplicados (dHash 64 bits). Limiar: distância de Hamming <= 6 -> a rever manualmente.
NEAR_DUP_T = 6
names_all = list(labeled_hash)
split_of = AUDIT.set_index("name").split.to_dict()
dups = []
for a, b in itertools.combinations(names_all, 2):
    d = hamming(labeled_hash[a], labeled_hash[b])
    if d <= NEAR_DUP_T:
        dups.append(dict(a=a, b=b, split_a=split_of[a], split_b=split_of[b], hamming=d))
DUPS = pd.DataFrame(dups, columns=["a", "b", "split_a", "split_b", "hamming"])
print(f"Pares de imagens anotadas com dHash <= {NEAR_DUP_T}: {len(DUPS)} | entre divisões diferentes: {int((DUPS.split_a != DUPS.split_b).sum()) if len(DUPS) else 0}")
if len(DUPS):
    display(DUPS)
print("NOTA: sem identificador de paciente não é possível garantir separação por paciente (várias imagens podem vir do mesmo doente);"
      " isto é uma limitação, partilhada com o artigo, que afeta a generalização estimada.")

# %%
# Imagens sem anotação: leitura robusta, extensão do conteúdo e exclusão de quase-duplicados de val/teste (anti-fuga no SSL)
UNL = []
EXCLUDED_UNL = []
if UNL_DIR is not None:
    heldout_hash = {n: labeled_hash[n] for n in SPLIT_NAMES["val"] + SPLIT_NAMES["test"] if n in labeled_hash}
    for p in sorted(UNL_DIR.glob("*.png")):
        try:
            img = read_rgb(p)
            h, w = content_extent(img)
            hsh = dhash(img[:h, :w])
            dmin, nearest = min((hamming(hsh, hv), hn) for hn, hv in heldout_hash.items())
            rec = dict(name=p.stem, path=p, h=h, w=w, dmin_heldout=dmin, nearest=nearest)
            (EXCLUDED_UNL if dmin <= CFG["ssl_dup_hamming"] else UNL).append(rec)
        except Exception as e:  # noqa: BLE001
            print("Não anotada ilegível:", p.name, e)
    print(f"Não anotadas: {len(UNL) + len(EXCLUDED_UNL)} | excluídas por quase-duplicado de val/teste (Hamming <= {CFG['ssl_dup_hamming']}): {len(EXCLUDED_UNL)} | usáveis: {len(UNL)}")
    pd.DataFrame(UNL + EXCLUDED_UNL).drop(columns=["path"]).to_csv(DIRS["tables"] / "audit_unlabeled.csv", index=False)
    if EXCLUDED_UNL:
        display(pd.DataFrame(EXCLUDED_UNL).drop(columns=["path"]).head(10))
else:
    print("Sem pasta Unlabeled: o SSL será omitido e o notebook segue em modo supervisionado.")
