# %% [markdown]
# ## 17–18. Comparação de experiências, *ensemble*, TTA e pós-processamento (Experiência D) — **só validação**
#
# Constrói-se um *pool* com os melhores checkpoints (supervisionados e SSL) e enumeram-se **candidatos**: melhor modelo único supervisionado,
# *ensembles* por arquitetura (sementes), melhor SSL, *ensemble* supervisionado completo e *ensemble* total. Para cada candidato escolhem-se na
# **validação** o TTA (`none`/`flip`/`flip_ms`) e a área mínima de componentes (com a mesma histerese de 0,002 para não ajustar ruído).
# As probabilidades são **médias uniformes** (afinar pesos numa validação de 16 imagens sobreajustaria). O candidato final é o de maior
# macro-Dice dos tecidos na validação; empates resolvem-se pelo menor nº de modelos.
#
# **O teste continua intocado nesta célula.** As probabilidades de validação são calculadas uma vez por (modelo, TTA) e reutilizadas.

# %%
POOL: dict[str, dict] = {}
for r in SUP:
    POOL[r["run_id"]] = dict(run=r, kind="sup", arch=r["spec"]["arch"], seed=r["spec"]["seed"], ckpt=r["ckpt"])
for sup_id, r in SSL_FINAL.items():
    if r["run_id"] != sup_id:                               # só entra se houve de facto SSL com ganho
        POOL[r["run_id"]] = dict(run=r, kind="ssl", arch=r["spec"]["arch"], seed=r["spec"]["seed"], ckpt=r["ckpt"])
SSL_IDS = [k for k, v in POOL.items() if v["kind"] == "ssl"]
SUP_IDS = [k for k, v in POOL.items() if v["kind"] == "sup"]
print(f"Pool: {len(SUP_IDS)} supervisionados, {len(SSL_IDS)} SSL (com ganho na validação)")

SIZE, FRAME = CFG["size"], CFG["frame"]
_recs = get_frame(FRAME)
VAL_ITEMS = EvalItems(_recs["val"], SIZE)
_MODEL_CACHE: dict[str, Net] = {}


def load_pool_model(rid: str) -> Net:
    if rid not in _MODEL_CACHE:
        m = Net(ARCH_SPECS[POOL[rid]["arch"]]["ctor"](None)).to(DEVICE).eval()
        m.load_state_dict(_load_half(Path(POOL[rid]["ckpt"])))
        _MODEL_CACHE[rid] = m
    return _MODEL_CACHE[rid]


_PROBS_VAL: dict[tuple, list] = {}


def val_probs(rid: str, tta: str) -> list[np.ndarray]:
    if (rid, tta) not in _PROBS_VAL:
        _PROBS_VAL[(rid, tta)] = predict_probs([load_pool_model(rid)], VAL_ITEMS, tta)
    return _PROBS_VAL[(rid, tta)]


def ens_probs(ids: list[str], tta: str, getter) -> list[np.ndarray]:
    per = [getter(i, tta) for i in ids]
    return [np.mean([p[j] for p in per], axis=0) for j in range(len(per[0]))]


def candidates() -> dict[str, list[str]]:
    c: dict[str, list[str]] = {}
    best_sup = max(SUP_IDS, key=lambda i: val_score(POOL[i]["run"]))
    c["sup_best_single"] = [best_sup]
    for a in CFG["archs"]:
        ids = [i for i in SUP_IDS if POOL[i]["arch"] == a]
        if len(ids) > 1:
            c[f"sup_seeds_{a}"] = ids
    per_arch_best = [max((i for i in SUP_IDS if POOL[i]["arch"] == a), key=lambda i: val_score(POOL[i]["run"])) for a in CFG["archs"] if any(POOL[i]["arch"] == a for i in SUP_IDS)]
    if len(per_arch_best) > 1:
        c["sup_best_per_arch"] = per_arch_best
    if len(SUP_IDS) > 1:
        c["sup_all"] = SUP_IDS
    if SSL_IDS:
        c["ssl_best_single"] = [max(SSL_IDS, key=lambda i: val_score(POOL[i]["run"]))]
        c["all_models"] = SUP_IDS + SSL_IDS
        ssl_arch_ids = [i for i in SUP_IDS if POOL[i]["arch"] == POOL[SSL_IDS[0]]["arch"]]
        c["ssl+sup_same_arch"] = list(dict.fromkeys(SSL_IDS + ssl_arch_ids))
    return {k: list(dict.fromkeys(v)) for k, v in c.items()}


CANDS = candidates()
rows, TUNED = [], {}
for name, ids in CANDS.items():
    if not BUDGET.ok(300):
        print("Orçamento de tempo: candidatos restantes ignorados."); break
    best = (-1, None)
    per_tta = {}
    for tta in CFG["tta_candidates"]:
        pr = ens_probs(ids, tta, val_probs)
        sc = evaluate_preds(VAL_ITEMS.names, VAL_ITEMS.gts, probs_to_preds(pr)).metrics()["macro_fg_dice"]
        per_tta[tta] = sc
        if sc > best[0] + 1e-9 and (best[1] is None or sc - best[0] > 0.002):   # histerese: só troca se ganhar > 0,002
            best = (sc, tta)
    tta = best[1] or "none"
    pr = ens_probs(ids, tta, val_probs)
    base_pp = (evaluate_preds(VAL_ITEMS.names, VAL_ITEMS.gts, probs_to_preds(pr, 0.0)).metrics()["macro_fg_dice"], 0.0)
    pp_best = base_pp
    for fr in CFG["pp_min_area_fracs"][1:]:
        sc = evaluate_preds(VAL_ITEMS.names, VAL_ITEMS.gts, probs_to_preds(pr, fr)).metrics()["macro_fg_dice"]
        if sc > pp_best[0] + 0.002:
            pp_best = (sc, fr)
    res = evaluate_preds(VAL_ITEMS.names, VAL_ITEMS.gts, probs_to_preds(pr, pp_best[1])).metrics()
    TUNED[name] = dict(ids=ids, tta=tta, pp=pp_best[1], val=res)
    rows.append(dict(candidato=name, n_modelos=len(ids), tta=tta, pp_min_area=pp_best[1], **{f"val_macro_fg(TTA={k})": round(v, 4) for k, v in per_tta.items()},
                     val_macro_fg=res["macro_fg_dice"], val_micro_fg=res["micro_fg_dice"], val_paper_dice=res["paper_dice"], val_fibrin=res["dice_fibrin"],
                     val_granulation=res["dice_granulation"], val_callus=res["dice_callus"]))
CAND_TABLE = pd.DataFrame(rows).sort_values(["val_macro_fg", "n_modelos"], ascending=[False, True]).reset_index(drop=True)
CAND_TABLE.to_csv(DIRS["tables"] / "candidates_val.csv", index=False)
print("Candidatos (VALIDAÇÃO; a seleção usa só estas colunas — o teste não foi lido):")
display(CAND_TABLE.round(4))
FINAL_NAME = CAND_TABLE.candidato.iloc[0]
FINAL = TUNED[FINAL_NAME]
print(f"\nCandidato final escolhido pela validação: {FINAL_NAME} | modelos={FINAL['ids']} | TTA={FINAL['tta']} | pp_min_area={FINAL['pp']}")
print("Nota: com 16 imagens de validação e vários candidatos, o valor de validação do vencedor é otimista (viés de seleção).")

# %% [markdown]
# ## 19. Avaliação final no **teste** (uma única vez)
#
# Cada acesso ao teste fica registado em `TEST_LEDGER` (modelo, TTA, utilização). Avaliam-se, **sem qualquer ajuste posterior**:
# (1) o melhor modelo único supervisionado (sem TTA nem pós-processamento) — comparável à fase SL do artigo;
# (2) o melhor SSL (se existir), sem TTA — comparável à fase SSL;
# (3) o candidato final escolhido na validação, com o seu TTA e pós-processamento. Os intervalos de confiança de 95 % são *bootstrap* sobre as imagens de teste.

# %%
TEST_ITEMS = EvalItems(_recs["test"], SIZE)
TEST_LEDGER: list[dict] = []
_PROBS_TEST: dict[tuple, list] = {}


def test_probs(rid: str, tta: str) -> list[np.ndarray]:
    if (rid, tta) not in _PROBS_TEST:
        TEST_LEDGER.append(dict(event="forward_on_test", model=rid, tta=tta, t=time.strftime("%H:%M:%S")))
        _PROBS_TEST[(rid, tta)] = predict_probs([load_pool_model(rid)], TEST_ITEMS, tta)
    return _PROBS_TEST[(rid, tta)]


def evaluate_test(label: str, ids: list[str], tta: str, pp: float) -> EvalResult:
    TEST_LEDGER.append(dict(event="evaluate", label=label, models=ids, tta=tta, pp=pp, t=time.strftime("%H:%M:%S")))
    pr = ens_probs(ids, tta, test_probs)
    return evaluate_preds(TEST_ITEMS.names, TEST_ITEMS.gts, probs_to_preds(pr, pp)), pr


TEST_ROWS: dict[str, dict] = {}
best_sup_id = CANDS["sup_best_single"][0]
spec_rows = [("Nosso: melhor supervisionado (1 modelo, sem TTA)", [best_sup_id], "none", 0.0)]
if "ssl_best_single" in CANDS:
    spec_rows.append(("Nosso: melhor SSL (1 modelo, sem TTA)", CANDS["ssl_best_single"], "none", 0.0))
spec_rows.append((f"Nosso: FINAL = {FINAL_NAME} (TTA={FINAL['tta']}, pp={FINAL['pp']})", FINAL["ids"], FINAL["tta"], FINAL["pp"]))
for label, ids, tta, pp in spec_rows:
    res, pr = evaluate_test(label, ids, tta, pp)
    TEST_ROWS[label] = dict(res=res, probs=pr, ids=ids, tta=tta, pp=pp, metrics=res.metrics())
FINAL_LABEL = spec_rows[-1][0]
print(f"Leituras do teste registadas: {len(TEST_LEDGER)} eventos")
display(pd.DataFrame(TEST_LEDGER).drop(columns=["t"], errors="ignore"))

# %% [markdown]
# ## 21. Tabela comparativa com o artigo
# * **Global (comparável ao artigo)** = `paper_*`: réplica do protocolo do código dos autores (ver secção 12). Mesmo assim, equivalência **não garantida**: dados de teste
#   iguais (16 imagens, divisão oficial), mas frame/protocolo exato do artigo, sementes e checkpoints são desconhecidos, e o teste é minúsculo.
# * `micro_fg` e `macro_fg` são as definições padrão, mais exigentes, e **não** são comparáveis ao 87,64 do artigo.
# * Por classe: `paper_dice_<c>` usa a convenção por imagem do código dos autores; `dice_<c>` é o Dice agregado sobre o dataset (padrão).

# %%
KEYS_CI = ["paper_dice", "micro_fg_dice", "macro_fg_dice", "paper_dice_fibrin", "paper_dice_granulation", "paper_dice_callus", "dice_fibrin", "dice_granulation", "dice_callus"]
for label, row in TEST_ROWS.items():
    row["ci"] = row["res"].bootstrap(KEYS_CI, CFG["bootstrap_b"], seed=0)


def fmt(m: dict, ci: dict | None, key: str) -> str:
    v = m[key]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    s = f"{100 * v:.2f}"
    if ci and key in ci:
        s += f" [{100 * ci[key][0]:.1f}–{100 * ci[key][1]:.1f}]"
    return s


cmp_rows = [
    dict(modelo="ARTIGO (reportado) — SL", dice_global_paper="84.89", IoU_paper="73.75", Precision_paper="87.69", Recall_paper="82.27", micro_fg="—", macro_fg="—", fibrin_paperconv="—", granulation_paperconv="—", callus_paperconv="—"),
    dict(modelo="ARTIGO (reportado) — SSL", dice_global_paper="87.64", IoU_paper="77.99", Precision_paper="88.14", Recall_paper="87.14", micro_fg="—", macro_fg="—",
         fibrin_paperconv="69.01", granulation_paperconv="94.11", callus_paperconv="78.27"),
]
for label, row in TEST_ROWS.items():
    m, ci = row["metrics"], row["ci"]
    cmp_rows.append(dict(modelo=label, dice_global_paper=fmt(m, ci, "paper_dice"), IoU_paper=fmt(m, None, "paper_iou"), Precision_paper=fmt(m, None, "paper_precision"),
                         Recall_paper=fmt(m, None, "paper_recall"), micro_fg=fmt(m, ci, "micro_fg_dice"), macro_fg=fmt(m, ci, "macro_fg_dice"),
                         fibrin_paperconv=fmt(m, ci, "paper_dice_fibrin"), granulation_paperconv=fmt(m, ci, "paper_dice_granulation"), callus_paperconv=fmt(m, ci, "paper_dice_callus")))
COMPARISON = pd.DataFrame(cmp_rows)
COMPARISON.to_csv(DIRS["tables"] / "final_comparison_test.csv", index=False)
print("TESTE (16 imagens) — valores em %, intervalo de confiança bootstrap 95% entre parênteses retos:")
display(COMPARISON)

cls_rows = []
for label, row in TEST_ROWS.items():
    m, ci = row["metrics"], row["ci"]
    for c in ["fibrin", "granulation", "callus"]:
        cls_rows.append(dict(modelo=label, classe=c, dice_agregado=fmt(m, ci, f"dice_{c}"), dice_convencao_artigo=fmt(m, ci, f"paper_dice_{c}"),
                             iou_agregado=fmt(m, None, f"iou_{c}"), precision_agregada=fmt(m, None, f"precision_{c}"), recall_agregado=fmt(m, None, f"recall_{c}"),
                             dice_artigo_SSL=PAPER["per_class_ssl"][c.capitalize()]["Dice"]))
display(pd.DataFrame(cls_rows))

# Veredicto calculado (nunca pré-preenchido)
fm, fci = TEST_ROWS[FINAL_LABEL]["metrics"], TEST_ROWS[FINAL_LABEL]["ci"]
verdict_rows = []
for key, ref, nm in [("paper_dice", 0.8764, "Dice global (protocolo-réplica) vs SSL do artigo 87,64"), ("paper_dice_fibrin", 0.6901, "Dice Fibrin (conv. artigo) vs 69,01"),
                     ("paper_dice_granulation", 0.9411, "Dice Granulation (conv. artigo) vs 94,11"), ("paper_dice_callus", 0.7827, "Dice Callus (conv. artigo) vs 78,27")]:
    v, lo, hi = fm[key], *fci[key]
    verdict_rows.append(dict(comparação=nm, nosso=round(100 * v, 2), IC95=f"[{100 * lo:.1f}, {100 * hi:.1f}]", artigo=round(100 * ref, 2),
                             ponto_acima=bool(v > ref), IC_exclui_artigo=bool(lo > ref or hi < ref)))
VERDICT = pd.DataFrame(verdict_rows)
display(VERDICT)
print("Leitura: 'ponto_acima' compara só a estimativa pontual; só se pode falar de melhoria sólida se o IC exclui o valor do artigo (e mesmo assim"
      " a comparação depende da equivalência do protocolo e de um teste de 16 imagens, possivelmente com várias imagens do mesmo doente).")
