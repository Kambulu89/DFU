# %% [markdown]
# ## 12. Métricas e testes automáticos
#
# Tudo é calculado a partir de **matrizes de confusão por imagem** (linhas = GT, colunas = predição), que permitem agregar sobre o dataset e fazer
# *bootstrap* sobre imagens. Definições (fundo explicitamente incluído/excluído):
#
# | Nome | Definição | Fundo |
# |---|---|---|
# | `dice_<classe>` | Dice da classe sobre a matriz de confusão **somada** no conjunto | — |
# | `macro_fg_dice` | média dos 3 Dice de tecido | **excluído** (métrica de seleção) |
# | `macro_all_dice` | média dos 4 Dice | incluído |
# | `micro_fg_dice` | 2ΣTP/(2ΣTP+ΣFP+ΣFN) sobre os 3 tecidos (confusões entre tecidos contam como FP *e* FN) | excluído |
# | `binary_fg_dice` | ferida (qualquer tecido) vs fundo | — |
# | `imgmean_dice_<classe>` | média por imagem, só onde a classe existe no GT ou na predição | — |
# | `paper_*` | **réplica do protocolo do código dos autores** (acumuladores não repostos entre classes; fundo excluído). É a única candidata a comparação direta com 87,64 | excluído |
#
# Os testes abaixo cobrem casos conhecidos, máscaras vazias, classes ausentes, desalinhamento e **concordância exata** da réplica
# vetorizada com uma porta literal do ciclo original (conjuntos de coordenadas) em casos aleatórios.

# %%
def confusion_matrix_np(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.shape != pred.shape:
        raise ValueError(f"Predição {pred.shape} desalinhada com o GT {gt.shape}")
    return np.bincount(gt.astype(np.int64).ravel() * K + pred.astype(np.int64).ravel(), minlength=K * K).reshape(K, K)


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def per_class_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    return dict(tp=tp, fp=fp, fn=fn, dice=_div(2 * tp, 2 * tp + fp + fn), iou=_div(tp, tp + fp + fn), precision=_div(tp, tp + fp), recall=_div(tp, tp + fn))


def paper_image_stats(cm: np.ndarray, ep: float = 1e-6) -> dict:
    """Réplica do ciclo por imagem do código dos autores (supervised_training.ipynb): acumuladores tp/fp/fn por imagem, **não repostos entre classes**,
    somados ao total global após cada classe comum; classes só em GT -> FN; só na predição -> FP; ausentes em ambas são ignoradas."""
    tp = fp = fn = 0.0
    c_tp = c_fp = c_fn = 0.0
    per = {}
    lg = {c for c in TISSUES if cm[c].sum() > 0}
    lp = {c for c in TISSUES if cm[:, c].sum() > 0}
    for d in sorted(lg - lp):
        fn += cm[d].sum()
        c_fn += fn
    for d in sorted(lp - lg):
        fp += cm[:, d].sum()
        c_fp += fp
    for d in sorted(lg ^ lp):
        per[d] = (0.0, 0.0, 0.0, 0.0)
    for c in sorted(lg & lp):
        t, f_p, f_n = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c].sum() - cm[c, c]
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
        c_tp, c_fp, c_fn = c_tp + tp, c_fp + fp, c_fn + fn
        per[c] = (tp / (tp + fp + ep) * 100, tp / (tp + fn + ep) * 100, 2 * tp / (2 * tp + fp + fn + ep) * 100, tp / (tp + fp + fn + ep) * 100)
    return dict(c=np.array([c_tp, c_fp, c_fn]), per=per)


def _paper_reference_literal(gt: np.ndarray, pred: np.ndarray, ep: float = 1e-6):
    """Porta LITERAL do ciclo original (conjuntos de coordenadas) — usada só para testar a versão vetorizada."""
    stp = sfp = sfn = 0
    tp = fp = fn = 0
    lbl_gt, lbl_pred = set(np.unique(gt)), set(np.unique(pred))
    lbl_gt.discard(0); lbl_pred.discard(0)
    diff1, diff2 = lbl_gt - lbl_pred, lbl_pred - lbl_gt
    diffs = diff1.union(diff2)
    per = {}
    for d1 in diff1:
        fn += len(np.argwhere(gt == d1)); sfn += fn
    for d2 in diff2:
        fp += len(np.argwhere(pred == d2)); sfp += fp
    for d in diffs:
        per[int(d)] = (0.0, 0.0, 0.0, 0.0)
    for cmn in lbl_gt.intersection(lbl_pred):
        gi, pi = np.where(gt == cmn), np.where(pred == cmn)
        g_t = tuple((gi[0][i], gi[1][i]) for i in range(len(gi[0])))
        p_t = tuple((pi[0][i], pi[1][i]) for i in range(len(pi[0])))
        tp += len(set(g_t) & set(p_t)); fp += len(set(p_t) - set(g_t)); fn += len(set(g_t) - set(p_t))
        stp += tp; sfp += fp; sfn += fn
        per[int(cmn)] = (tp / (tp + fp + ep) * 100, tp / (tp + fn + ep) * 100, 2 * tp / (2 * tp + fp + fn + ep) * 100, tp / (tp + fp + fn + ep) * 100)
    return np.array([stp, sfp, sfn], float), per


class EvalResult:
    """Resultado de avaliação sobre N imagens: guarda matrizes de confusão por imagem e estatísticas do protocolo-réplica."""

    def __init__(self, names: list[str], cms: np.ndarray):
        self.names, self.cms = list(names), np.asarray(cms, dtype=np.int64)
        ps = [paper_image_stats(cm) for cm in self.cms]
        self.paper_c = np.stack([p["c"] for p in ps])                      # (N,3): contribuições a stp, sfp, sfn
        self.paper_per = np.full((len(ps), K, 4), np.nan)                  # (N,K,4): P,R,Dice,IoU por classe (nan = não avaliada)
        for i, p in enumerate(ps):
            for c, v in p["per"].items():
                self.paper_per[i, c] = v

    def __len__(self):
        return len(self.names)

    def metrics(self, idx=None) -> dict:
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        cm = self.cms[idx].sum(0)
        pc = per_class_from_cm(cm)
        fg = TISSUES
        tp, fp, fn = pc["tp"][fg].sum(), pc["fp"][fg].sum(), pc["fn"][fg].sum()
        tb, fb, nb = cm[1:, 1:].sum(), cm[0, 1:].sum(), cm[1:, 0].sum()
        c = self.paper_c[idx].sum(0)
        pp = self.paper_per[idx]
        out = dict(
            macro_fg_dice=np.nanmean(pc["dice"][fg]) if not np.isnan(pc["dice"][fg]).all() else np.nan,
            macro_all_dice=np.nanmean(pc["dice"]) if not np.isnan(pc["dice"]).all() else np.nan,
            macro_fg_iou=np.nanmean(pc["iou"][fg]) if not np.isnan(pc["iou"][fg]).all() else np.nan,
            macro_all_iou=np.nanmean(pc["iou"]) if not np.isnan(pc["iou"]).all() else np.nan,
            macro_fg_precision=np.nanmean(pc["precision"][fg]) if not np.isnan(pc["precision"][fg]).all() else np.nan,
            macro_fg_recall=np.nanmean(pc["recall"][fg]) if not np.isnan(pc["recall"][fg]).all() else np.nan,
            micro_fg_dice=_div(2 * tp, 2 * tp + fp + fn), micro_fg_iou=_div(tp, tp + fp + fn),
            micro_fg_precision=_div(tp, tp + fp), micro_fg_recall=_div(tp, tp + fn),
            binary_fg_dice=_div(2 * tb, 2 * tb + fb + nb), pixel_acc=_div(np.trace(cm), cm.sum()),
            paper_dice=_div(2 * c[0], 2 * c[0] + c[1] + c[2]), paper_iou=_div(c[0], c.sum()),
            paper_precision=_div(c[0], c[0] + c[1]), paper_recall=_div(c[0], c[0] + c[2]),
        )
        out = {k: float(v) for k, v in out.items()}
        for k in range(K):
            n = CLASS_NAMES[k].lower()
            for key in ("dice", "iou", "precision", "recall"):
                out[f"{key}_{n}"] = float(pc[key][k])
            if k in fg:
                col = pp[:, k, :]
                ok = ~np.isnan(col[:, 2])
                for j, key in enumerate(("precision", "recall", "dice", "iou")):
                    out[f"paper_{key}_{n}"] = float(col[ok, j].mean() / 100) if ok.any() else np.nan
        # média por imagem (só onde a classe existe em GT ou predição)
        for k in fg:
            d = np.array([per_class_from_cm(m)["dice"][k] for m in self.cms[idx]])
            out[f"imgmean_dice_{CLASS_NAMES[k].lower()}"] = float(np.nanmean(d)) if (~np.isnan(d)).any() else np.nan
        return out

    def bootstrap(self, keys: list[str], B: int, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        draws = {k: [] for k in keys}
        n = len(self)
        for _ in range(B):
            m = self.metrics(rng.integers(0, n, n))
            for k in keys:
                draws[k].append(m[k])
        return {k: tuple(np.nanpercentile(v, [2.5, 97.5])) for k, v in draws.items()}


def evaluate_preds(names: list[str], gts: list[np.ndarray], preds: list[np.ndarray]) -> EvalResult:
    if not (len(names) == len(gts) == len(preds)):
        raise ValueError("Número de GT/predições/nomes inconsistente")
    return EvalResult(names, np.stack([confusion_matrix_np(g, p) for g, p in zip(gts, preds)]))


def run_metric_tests() -> None:
    rng = np.random.default_rng(123)
    # 1) exemplo conhecido
    gt = np.array([[0, 1], [2, 2]]); pr = np.array([[0, 1], [2, 1]])
    m = evaluate_preds(["a"], [gt], [pr]).metrics()
    assert abs(m["dice_fibrin"] - 2 / 3) < 1e-9 and abs(m["dice_granulation"] - 2 / 3) < 1e-9 and np.isnan(m["dice_callus"])
    assert abs(m["macro_fg_dice"] - 2 / 3) < 1e-9 and abs(m["micro_fg_dice"] - 2 / 3) < 1e-9
    # 2) predição perfeita -> 1 em tudo
    g = rng.integers(0, 4, (30, 30)); m = evaluate_preds(["p"], [g], [g.copy()]).metrics()
    assert m["macro_fg_dice"] == 1 and m["micro_fg_dice"] == 1 and m["paper_dice"] == 1 and m["pixel_acc"] == 1
    # 3) máscaras vazias (só fundo) e predição vazia: classes ausentes em ambos -> NaN, sem divisão por zero/exceção
    z = np.zeros((16, 16), int); m = evaluate_preds(["e"], [z], [z]).metrics()
    assert np.isnan(m["macro_fg_dice"]) and np.isnan(m["paper_dice"]) and m["pixel_acc"] == 1
    # 4) classe só no GT (não prevista) -> Dice 0; só na predição -> Dice 0 (FP)
    gt = np.zeros((10, 10), int); gt[:5] = 1; m = evaluate_preds(["f"], [gt], [np.zeros_like(gt)]).metrics()
    assert m["dice_fibrin"] == 0 and m["recall_fibrin"] == 0
    m = evaluate_preds(["g"], [np.zeros_like(gt)], [gt]).metrics()
    assert m["dice_fibrin"] == 0 and m["precision_fibrin"] == 0
    # 5) desalinhamento deve falhar de forma segura
    try:
        confusion_matrix_np(np.zeros((4, 4), int), np.zeros((4, 5), int)); raise AssertionError("devia falhar")
    except ValueError:
        pass
    # 6) concordância com sklearn
    from sklearn.metrics import confusion_matrix as sk_cm
    g, p = rng.integers(0, 4, (50, 50)), rng.integers(0, 4, (50, 50))
    assert (confusion_matrix_np(g, p) == sk_cm(g.ravel(), p.ravel(), labels=range(K))).all()
    # 7) réplica vetorizada == porta literal do ciclo original (inclui classes ausentes/só-GT/só-predição)
    for t in range(150):
        g = rng.integers(0, 4, (12, 12)) * (rng.random((12, 12)) < rng.uniform(0.2, 1))
        p = np.where(rng.random((12, 12)) < rng.uniform(0.3, 1), g, rng.integers(0, 4, (12, 12)))
        if t % 5 == 0:
            g[g == 2] = 0
        lit_c, lit_per = _paper_reference_literal(g, p)
        vec = paper_image_stats(confusion_matrix_np(g, p))
        assert np.allclose(lit_c, vec["c"]), (t, lit_c, vec["c"])
        assert lit_per.keys() == vec["per"].keys() and all(np.allclose(lit_per[k], vec["per"][k]) for k in lit_per), (t, lit_per, vec["per"])
    # 8) bootstrap: IC contém a estimativa pontual em dados simulados
    g = [rng.integers(0, 4, (20, 20)) for _ in range(12)]; p = [np.where(rng.random((20, 20)) < 0.7, x, rng.integers(0, 4, (20, 20))) for x in g]
    r = evaluate_preds([str(i) for i in range(12)], g, p); ci = r.bootstrap(["macro_fg_dice"], 200)["macro_fg_dice"]
    assert ci[0] <= r.metrics()["macro_fg_dice"] <= ci[1]
    print("OK: testes de métricas (exemplo conhecido, perfeito, vazias, ausentes, desalinhamento, sklearn, réplica==porta literal em 150 casos, bootstrap)")


run_metric_tests()
