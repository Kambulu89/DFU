# %% [markdown]
# ## 15. Infraestrutura de avaliação: previsões fora de amostra (OOF), variantes e bootstrap pareado
#
# * **Membro** = um modelo treinado (um fold × arquitetura × semente, supervisionado ou SSL). **Variante** = todos os membros do mesmo tipo e arquitetura
#   (p.ex. `sup:upernet_convnext_s`, `ssl0:segformer_mitb3`).
# * **Conjunto de avaliação de desenvolvimento** (`EVAL_*`): em CV são as **94 imagens de desenvolvimento**; no modo oficial são as 16 de validação.
#   A previsão de uma variante para uma imagem é a média das probabilidades dos membros que **não treinaram nessa imagem** (em CV: exatamente o modelo do fold que a reteve).
#   Só variantes que cobrem **todas** as imagens de avaliação são elegíveis (se o orçamento de tempo cortou folds, a variante é descartada e isso é indicado).
# * **Bootstrap pareado**: reamostra as mesmas imagens para duas configurações e devolve Δ, IC 95 % e P(Δ>0) — é a ferramenta para decidir se uma melhoria (SSL, ensemble,
#   TTA, pós-processamento) é real. Com 94 imagens tem muito mais poder do que com 16.
# * Nada aqui toca no teste.

# %%
SIZE, FRAME = CFG["size"], CFG["frame"]
_recs = get_frame(FRAME)
EVAL_RECS = (_recs["train"] + _recs["val"]) if CFG["do_cv"] else _recs["val"]
EVAL_ITEMS = EvalItems(EVAL_RECS, SIZE)
EVAL_INDEX = {n: j for j, n in enumerate(EVAL_ITEMS.names)}
assert not (set(EVAL_INDEX) & set(SPLIT_NAMES["test"])), "Fuga: o conjunto de avaliação de desenvolvimento contém imagens de teste"
print(f"Conjunto de avaliação de desenvolvimento: {len(EVAL_ITEMS)} imagens ({'OOF por validação cruzada' if CFG['do_cv'] else 'validação oficial'}); frame={FRAME}, tela={SIZE}")

POOL: dict[str, dict] = {}
VARIANTS: dict[str, list[str]] = {}


def register_member(run: dict, kind: str) -> None:
    s, rid = run["spec"], run["run_id"]
    assert s["size"] == SIZE and s["frame"] == FRAME, f"{rid}: frame/tela ({s['frame']}/{s['size']}) diferem dos de avaliação ({FRAME}/{SIZE})"
    POOL[rid] = dict(run=run, kind=kind, arch=s["arch"], seed=s["seed"], fold=s.get("fold"), ckpt=run["ckpt"], val_names=run["val_names"])
    if rid not in VARIANTS.setdefault(f"{kind}:{s['arch']}", []):
        VARIANTS[f"{kind}:{s['arch']}"].append(rid)


for _r in SUP:
    register_member(_r, "sup")

_MODEL_CACHE: "OrderedDict[str, Net]" = OrderedDict()
MODEL_CACHE_MAX = 6   # o pool pode ter dezenas de modelos: mantém-se só os mais recentes na GPU (as probabilidades já calculadas ficam em cache à parte)


def load_pool_model(rid: str) -> Net:
    if rid in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(rid)
        return _MODEL_CACHE[rid]
    m = Net(ARCH_SPECS[POOL[rid]["arch"]]["ctor"](None)).to(DEVICE).eval()
    m.load_state_dict(_load_half(Path(POOL[rid]["ckpt"])))
    _MODEL_CACHE[rid] = m
    while len(_MODEL_CACHE) > MODEL_CACHE_MAX:
        _MODEL_CACHE.popitem(last=False)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return m


_MEMBER_ITEMS: dict[str, EvalItems] = {}
_MEMBER_PROBS: dict[tuple, dict[str, np.ndarray]] = {}


def member_probs(rid: str, tta: str) -> dict[str, np.ndarray]:
    """Probabilidades do membro nas imagens que ele NÃO viu no treino (dentro do conjunto de avaliação): {nome: (K,h,w)}."""
    if (rid, tta) not in _MEMBER_PROBS:
        if rid not in _MEMBER_ITEMS:
            held = set(POOL[rid]["val_names"])
            _MEMBER_ITEMS[rid] = EvalItems([r for r in EVAL_RECS if r["name"] in held], SIZE)
        items = _MEMBER_ITEMS[rid]
        pr = predict_probs([load_pool_model(rid)], items, tta)
        _MEMBER_PROBS[(rid, tta)] = dict(zip(items.names, pr))
    return _MEMBER_PROBS[(rid, tta)]


def variant_coverage(vname: str) -> float:
    covered = set()
    for rid in VARIANTS[vname]:
        covered |= set(POOL[rid]["val_names"]) & set(EVAL_INDEX)
    return len(covered) / len(EVAL_INDEX)


def complete_variants() -> list[str]:
    ok = [v for v in VARIANTS if variant_coverage(v) >= 0.999]
    for v in VARIANTS:
        if v not in ok:
            print(f"  variante descartada por cobertura incompleta ({100 * variant_coverage(v):.0f} % das imagens de avaliação): {v}")
    return ok


def variant_probs(vname: str, tta: str) -> list[np.ndarray]:
    per = [member_probs(rid, tta) for rid in VARIANTS[vname]]
    out = []
    for n in EVAL_ITEMS.names:
        ps = [d[n] for d in per if n in d]
        out.append(np.mean(ps, axis=0))
    return out


def cand_probs(variants: list[str], tta: str) -> list[np.ndarray]:
    per = [variant_probs(v, tta) for v in variants]
    return [np.mean([p[j] for p in per], axis=0) for j in range(len(EVAL_ITEMS))]


def eval_dev(probs: list[np.ndarray], pp=0.0, idx=None) -> EvalResult:
    idx = list(range(len(EVAL_ITEMS))) if idx is None else list(idx)
    return evaluate_preds([EVAL_ITEMS.names[i] for i in idx], [EVAL_ITEMS.gts[i] for i in idx], probs_to_preds([probs[i] for i in idx], pp))


def dev_score(probs, pp=0.0, idx=None) -> float:
    v = eval_dev(probs, pp, idx).metrics(light=True)["macro_fg_dice"]
    return 0.0 if np.isnan(v) else v


def tune_postproc(probs: list[np.ndarray], idx=None) -> tuple[dict, float, float]:
    """Ajusta, SÓ no conjunto de desenvolvimento (OOF), (1) escala de probabilidade por classe (a priori), (2) área mínima global e (3) área mínima da fibrina e (4) confiança mínima dos componentes de fibrina,
    por ascensão de coordenadas; cada passo só é aceite se ganhar > `pp_gain_margin` (histerese contra sobreajuste). Devolve (pp, score_base, score_final)."""
    margin = CFG["pp_gain_margin"]
    pp = norm_pp(None)
    base = best = dev_score(probs, pp, idx)
    if CFG["tune_class_scale"]:
        for _ in range(2):
            for c in TISSUES:
                for s in CFG["pp_scale_grid"]:
                    if s == pp["scale"][c]:
                        continue
                    cand = {**pp, "scale": [s if i == c else v for i, v in enumerate(pp["scale"])]}
                    sc = dev_score(probs, cand, idx)
                    if sc > best + margin:
                        pp, best = cand, sc
    for key, grid in [("min_frac", CFG["pp_min_area_fracs"]), ("fibrin_frac", CFG["pp_fibrin_fracs"]), ("fibrin_conf", CFG["pp_fibrin_conf"])]:
        for g in grid:
            if g == pp[key]:
                continue
            cand = {**pp, key: g}
            sc = dev_score(probs, cand, idx)
            if sc > best + margin:
                pp, best = cand, sc
    return pp, base, best


def pp_honest_gain(probs: list[np.ndarray], reps: int = 10, seed: int = 0) -> tuple[float, float]:
    """Ganho do pós-processamento ajustado numa metade das imagens e medido na OUTRA metade (média de `reps` divisões aleatórias): estima o viés
    de ajustar e avaliar nas mesmas imagens. Devolve (ganho médio, desvio) em macro-Dice dos tecidos."""
    rng, gains = np.random.default_rng(seed), []
    n = len(EVAL_ITEMS)
    for _ in range(reps):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        pp, _, _ = tune_postproc(probs, a)
        gains.append(dev_score(probs, pp, b) - dev_score(probs, 0.0, b))
    return float(np.mean(gains)), float(np.std(gains))


def paired_table(pairs: list[tuple[str, EvalResult, EvalResult]], keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or ["macro_fg_dice", "paper_dice", "dice_fibrin", "dice_granulation", "dice_callus"]
    rows = []
    for label, a, b in pairs:
        pb = paired_bootstrap(a, b, keys, CFG["paired_bootstrap_b"], seed=1)
        for k in keys:
            d = pb[k]
            rows.append(dict(comparação=label, métrica=k, delta_pp=round(100 * d["delta"], 2), IC95=f"[{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]", P_delta_positivo=round(d["p_gt0"], 3),
                             sinal_robusto=("A>B" if d["lo"] > 0 else "A<B" if d["hi"] < 0 else "indistinguível")))
    return pd.DataFrame(rows)


# %%
# Desempenho OOF das variantes supervisionadas (conjunto de desenvolvimento; NÃO é o teste)
SUP_VARIANTS = [v for v in complete_variants() if v.startswith("sup:")]
assert SUP_VARIANTS, "Nenhuma variante supervisionada completa: aumente o orçamento de tempo ou reduza folds/arquiteturas"
OOF_SUP = {v: eval_dev(variant_probs(v, "none")) for v in SUP_VARIANTS}
rows = []
for v, res in OOF_SUP.items():
    m, ci = res.metrics(), res.bootstrap(["macro_fg_dice", "dice_fibrin"], CFG["bootstrap_b"] // 2, seed=0)
    rows.append(dict(variante=v, n_membros=len(VARIANTS[v]), macro_fg=m["macro_fg_dice"], macro_fg_IC95=f"[{ci['macro_fg_dice'][0]:.3f}, {ci['macro_fg_dice'][1]:.3f}]", micro_fg=m["micro_fg_dice"],
                     paper_dice=m["paper_dice"], fibrin=m["dice_fibrin"], granulation=m["dice_granulation"], callus=m["dice_callus"]))
OOF_TABLE = pd.DataFrame(rows).sort_values("macro_fg", ascending=False).reset_index(drop=True)
OOF_TABLE.to_csv(DIRS["tables"] / "oof_supervised_variants.csv", index=False)
print(f"Desempenho {'OOF (94 imagens)' if CFG['do_cv'] else 'na validação oficial (16 imagens)'} das variantes supervisionadas:")
display(OOF_TABLE.round(4))
if len(OOF_SUP) > 1:
    best_v = OOF_TABLE.variante.iloc[0]
    pt = paired_table([(f"{best_v} vs {o}", OOF_SUP[best_v], OOF_SUP[o]) for o in OOF_TABLE.variante.iloc[1:]], ["macro_fg_dice", "dice_fibrin", "dice_callus"])
    print("Comparação pareada (bootstrap sobre imagens) da melhor variante contra as restantes:")
    display(pt)
    pt.to_csv(DIRS["tables"] / "paired_sup_variants.csv", index=False)
RANKED_ARCHS = [v.split(":", 1)[1] for v in OOF_TABLE.variante]
print("Ranking de arquiteturas (OOF, sem TTA):", RANKED_ARCHS)
