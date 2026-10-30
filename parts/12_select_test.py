# %% [markdown]
# ## 17–18. Comparação de experiências, *ensemble*, TTA e pós-processamento (Experiência D) — **só desenvolvimento (OOF/validação)**
#
# Enumeram-se **candidatos** (cada variante isolada, *ensemble* de todas as supervisionadas, das duas melhores, por tipo de SSL e de tudo). Para cada candidato escolhe-se
# **no conjunto de desenvolvimento** o TTA (`none`/`flip`/`flip_ms`, com histerese) e **o candidato final é o de maior macro-Dice dos tecidos com esse TTA** (empate → menos modelos).
# As probabilidades são **médias uniformes** (afinar pesos de *ensemble* sobreajustaria).
#
# **Pós-processamento (escala de probabilidade por classe, área mínima global e da fibrina) com porta de segurança.** É ajustado **apenas para o candidato vencedor** e só é **aplicado se o ganho
# "honesto" for superior a 0,2 pp** (ajustado numa metade das imagens e avaliado na outra; média de 10 repetições). Na corrida anterior esse ganho foi −0,24 pp: o ajuste era sobreajuste, e é isso que a
# porta evita. **O teste continua intocado nesta secção.**

# %%
ALL_VARS = complete_variants()
SUP_V = [v for v in ALL_VARS if v.startswith("sup:")]
SSL_V = [v for v in ALL_VARS if v.startswith("ssl")]


def make_candidates() -> dict[str, list[str]]:
    c = {v: [v] for v in ALL_VARS}
    if len(SUP_V) > 1:
        c["sup_all"] = list(SUP_V)
    if len(SUP_V) > 2:
        c["sup_top2"] = [f"sup:{a}" for a in RANKED_ARCHS[:2] if f"sup:{a}" in SUP_V]
    for rnd in sorted({v.split(":")[0] for v in SSL_V}):
        vs = [v for v in SSL_V if v.startswith(rnd + ":")]
        if len(vs) > 1:
            c[f"{rnd}_all"] = vs
    if SSL_V:
        c["sup+ssl_all"] = SUP_V + SSL_V
    return {k: list(dict.fromkeys(v)) for k, v in c.items()}


def members_of(variants: list[str]) -> list[str]:
    return list(dict.fromkeys(rid for v in variants for rid in VARIANTS[v]))


CANDS = make_candidates()
rows, TUNED = [], {}
for name, vs in CANDS.items():          # sem verificação de tempo: o orçamento reserva tempo para esta fase (numa corrida, cortá-la deitou fora 7 h de SSL)
    per_tta, best_tta = {}, (-1.0, None)
    for tta in CFG["tta_candidates"]:
        sc = dev_score(cand_probs(vs, tta))
        per_tta[tta] = sc
        if best_tta[1] is None or sc - best_tta[0] > 0.002:      # histerese: só troca de TTA se ganhar > 0,002
            best_tta = (sc, tta)
    tta = best_tta[1]
    pr = cand_probs(vs, tta)
    res = eval_dev(pr, 0.0).metrics()
    TUNED[name] = dict(variants=vs, members=members_of(vs), tta=tta, pp=norm_pp(None), dev=res, dev_raw=per_tta["none"])
    rows.append(dict(candidato=name, n_modelos=len(TUNED[name]["members"]), tta=tta, dev_objetivo_sem_tta=per_tta["none"], dev_objetivo=objective(res), dev_macro_fg=res["macro_fg_dice"], dev_micro_fg=res["micro_fg_dice"],
                     dev_paper_dice=res["paper_dice"], dev_fibrin=res["dice_fibrin"], dev_granulation=res["dice_granulation"], dev_callus=res["dice_callus"]))
CAND_TABLE = pd.DataFrame(rows).sort_values(["dev_objetivo", "n_modelos"], ascending=[False, True]).reset_index(drop=True)
CAND_TABLE.to_csv(DIRS["tables"] / "candidates_dev.csv", index=False)
print(f"Objetivo das decisões: {CFG['objective']} (macro = macro-Dice dos tecidos; paper = Dice global do artigo; combo = média). Candidatos ({'OOF, 94 imagens' if CFG['do_cv'] else 'validação oficial, 16 imagens'}; a seleção usa só estas colunas — o teste não foi lido):")
display(CAND_TABLE.round(4))
FINAL_NAME = CAND_TABLE.candidato.iloc[0]
FINAL = TUNED[FINAL_NAME]
print(f"\nCandidato final escolhido: {FINAL_NAME} | {len(FINAL['members'])} modelos | TTA={FINAL['tta']}")
print("Nota: o valor de desenvolvimento do vencedor é otimista (escolha entre vários candidatos e do TTA nas mesmas imagens).")

# pós-processamento só para o vencedor, com porta de segurança (ganho honesto > 0)
_probs_final = cand_probs(FINAL["variants"], FINAL["tta"])
_pp_try, _sc0, _sc1 = tune_postproc(_probs_final)
PP_HONEST_GAIN = pp_honest_gain(_probs_final)
PP_ACCEPTED = PP_HONEST_GAIN[0] > CFG["pp_gain_margin"] and _sc1 > _sc0      # ganho honesto médio > 0,2 pp
print(f"Pós-processamento candidato: {_pp_try} | ganho no próprio desenvolvimento: {100 * (_sc1 - _sc0):+.2f} pp | ganho HONESTO (metade/metade, 10 reps): {100 * PP_HONEST_GAIN[0]:+.2f} ± {100 * PP_HONEST_GAIN[1]:.2f} pp"
      f" -> {'ACEITE' if PP_ACCEPTED else 'REJEITADO (sem ganho honesto): o modelo final usa só o TTA'}")
FINAL["pp"] = _pp_try if PP_ACCEPTED else norm_pp(None)

# %%
# Efeito do TTA/pós-processamento e do ensemble sobre o melhor supervisionado — medidos no desenvolvimento com bootstrap pareado
best_sup_v = OOF_TABLE.variante.iloc[0]
raw_final = eval_dev(cand_probs(FINAL["variants"], "none"))
tuned_final = eval_dev(cand_probs(FINAL["variants"], FINAL["tta"]), FINAL["pp"])
pairs = [(f"FINAL (TTA{' + pós-proc.' if PP_ACCEPTED else ''}) vs FINAL sem TTA", tuned_final, raw_final)]
if best_sup_v not in FINAL["variants"] or len(FINAL["variants"]) > 1:
    pairs.append((f"FINAL vs {best_sup_v} (sem TTA)", tuned_final, OOF_SUP[best_sup_v]))
DEV_PAIRED = paired_table(pairs, ["macro_fg_dice", "paper_dice", "dice_fibrin", "dice_callus"])
display(DEV_PAIRED)
DEV_PAIRED.to_csv(DIRS["tables"] / "paired_final_dev.csv", index=False)

# %% [markdown]
# ## 19. Avaliação final no **teste** (uma única vez)
#
# Cada acesso ao teste fica registado em `TEST_LEDGER` (modelo, TTA, utilização). Avaliam-se, **sem qualquer ajuste posterior**:
# (1) a melhor variante supervisionada (sem TTA nem pós-processamento) — comparável à fase SL do artigo;
# (2) a melhor variante SSL (se existir), sem TTA — comparável à fase SSL;
# (3) o candidato final escolhido no desenvolvimento, com o seu TTA e pós-processamento. Cada variante é o *ensemble* dos seus modelos (em CV: um por fold).
# Os intervalos de confiança de 95 % são *bootstrap* sobre as imagens de teste; as comparações entre linhas usam bootstrap **pareado**.

# %%
TEST_ITEMS = EvalItems(_recs["test"], SIZE)
TEST_LEDGER: list[dict] = []
_PROBS_TEST: dict[tuple, list] = {}


def test_probs(rid: str, tta: str) -> list[np.ndarray]:
    if (rid, tta) not in _PROBS_TEST:
        TEST_LEDGER.append(dict(event="forward_on_test", model=rid, tta=tta, t=time.strftime("%H:%M:%S")))
        _PROBS_TEST[(rid, tta)] = predict_probs([load_pool_model(rid)], TEST_ITEMS, tta)
    return _PROBS_TEST[(rid, tta)]


def evaluate_test(label: str, members: list[str], tta: str, pp) -> tuple[EvalResult, list[np.ndarray]]:
    TEST_LEDGER.append(dict(event="evaluate", label=label, models=members, tta=tta, pp=pp, t=time.strftime("%H:%M:%S")))
    per = [test_probs(rid, tta) for rid in members]
    pr = [np.mean([p[j] for p in per], axis=0) for j in range(len(TEST_ITEMS))]
    return evaluate_preds(TEST_ITEMS.names, TEST_ITEMS.gts, probs_to_preds(pr, pp)), pr


TEST_ROWS: dict[str, dict] = {}
spec_rows = [(f"Nosso: melhor supervisionado = {best_sup_v} (ensemble de {len(VARIANTS[best_sup_v])}, sem TTA)", VARIANTS[best_sup_v], "none", 0.0)]
ssl_ranked = sorted(SSL_V, key=lambda v: -eval_dev(variant_probs(v, "none")).metrics(light=True)["macro_fg_dice"]) if SSL_V else []
if ssl_ranked:
    spec_rows.append((f"Nosso: melhor SSL = {ssl_ranked[0]} (ensemble de {len(VARIANTS[ssl_ranked[0]])}, sem TTA)", VARIANTS[ssl_ranked[0]], "none", 0.0))
spec_rows.append((f"Nosso: FINAL = {FINAL_NAME} (TTA={FINAL['tta']}{', pp aceite' if PP_ACCEPTED else ''})", FINAL["members"], FINAL["tta"], FINAL["pp"]))
for label, members, tta, pp in spec_rows:
    res, pr = evaluate_test(label, members, tta, pp)
    TEST_ROWS[label] = dict(res=res, probs=pr, members=members, tta=tta, pp=pp, metrics=res.metrics())
FINAL_LABEL = spec_rows[-1][0]
print(f"Leituras do teste registadas: {len(TEST_LEDGER)} eventos")
display(pd.DataFrame(TEST_LEDGER).drop(columns=["t", "models"], errors="ignore"))

# %% [markdown]
# ## 21. Tabela comparativa com o artigo
# * **Global (comparável ao artigo)** = `paper_*`: réplica do protocolo do código dos autores (ver secção 12). Equivalência **não garantida**: teste igual (16 imagens, divisão oficial), mas o frame/protocolo exato do
#   artigo, sementes e *checkpoints* são desconhecidos, e o teste é minúsculo.
# * `micro_fg` e `macro_fg` são as definições padrão, mais exigentes, e **não** são comparáveis ao 87,64 do artigo.
# * Por classe: o **Dice agregado padrão** (`dice_<c>`) é a comparação mais plausível com o artigo; a "réplica por classe" (`paper_dice_<c>`) herda os acumuladores não repostos dos autores e fica só por transparência.

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
    dict(modelo="ARTIGO (reportado) — SL", dice_global_paper="84.89", IoU_paper="73.75", Precision_paper="87.69", Recall_paper="82.27", micro_fg="—", macro_fg="—", fibrin_agregado="—", granulation_agregado="—", callus_agregado="—"),
    dict(modelo="ARTIGO (reportado) — SSL", dice_global_paper="87.64", IoU_paper="77.99", Precision_paper="88.14", Recall_paper="87.14", micro_fg="—", macro_fg="—",
         fibrin_agregado="69.01", granulation_agregado="94.11", callus_agregado="78.27"),
]
for label, row in TEST_ROWS.items():
    m, ci = row["metrics"], row["ci"]
    cmp_rows.append(dict(modelo=label, dice_global_paper=fmt(m, ci, "paper_dice"), IoU_paper=fmt(m, None, "paper_iou"), Precision_paper=fmt(m, None, "paper_precision"),
                         Recall_paper=fmt(m, None, "paper_recall"), micro_fg=fmt(m, ci, "micro_fg_dice"), macro_fg=fmt(m, ci, "macro_fg_dice"),
                         fibrin_agregado=fmt(m, ci, "dice_fibrin"), granulation_agregado=fmt(m, ci, "dice_granulation"), callus_agregado=fmt(m, ci, "dice_callus")))
COMPARISON = pd.DataFrame(cmp_rows)
COMPARISON.to_csv(DIRS["tables"] / "final_comparison_test.csv", index=False)
print("TESTE (16 imagens) — valores em %, intervalo de confiança bootstrap 95% entre parênteses retos:")
display(COMPARISON)

cls_rows = []
for label, row in TEST_ROWS.items():
    m, ci = row["metrics"], row["ci"]
    for c in ["fibrin", "granulation", "callus"]:
        cls_rows.append(dict(modelo=label, classe=c, dice_agregado=fmt(m, ci, f"dice_{c}"), dice_replica_por_classe=fmt(m, ci, f"paper_dice_{c}"),
                             iou_agregado=fmt(m, None, f"iou_{c}"), precision_agregada=fmt(m, None, f"precision_{c}"), recall_agregado=fmt(m, None, f"recall_{c}"),
                             dice_artigo_SSL=PAPER["per_class_ssl"][c.capitalize()]["Dice"]))
display(pd.DataFrame(cls_rows))

# Veredicto calculado (nunca pré-preenchido)
fm, fci = TEST_ROWS[FINAL_LABEL]["metrics"], TEST_ROWS[FINAL_LABEL]["ci"]
verdict_rows = []
for key, ref, nm in [("paper_dice", 0.8764, "Dice global (protocolo-réplica) vs SSL do artigo 87,64"), ("paper_dice", 0.8489, "Dice global (protocolo-réplica) vs SL do artigo 84,89"),
                     ("dice_fibrin", 0.6901, "Dice Fibrin (agregado padrão) vs 69,01"), ("dice_granulation", 0.9411, "Dice Granulation (agregado padrão) vs 94,11"),
                     ("dice_callus", 0.7827, "Dice Callus (agregado padrão) vs 78,27"),
                     ("paper_dice_fibrin", 0.6901, "[réplica por classe, ver nota] Fibrin vs 69,01"), ("paper_dice_granulation", 0.9411, "[réplica por classe, ver nota] Granulation vs 94,11"),
                     ("paper_dice_callus", 0.7827, "[réplica por classe, ver nota] Callus vs 78,27")]:
    v, lo, hi = fm[key], *fci[key]
    verdict_rows.append(dict(comparação=nm, nosso=round(100 * v, 2), IC95=f"[{100 * lo:.1f}, {100 * hi:.1f}]", artigo=round(100 * ref, 2),
                             ponto_acima=bool(v > ref), IC_exclui_artigo=bool(lo > ref or hi < ref)))
VERDICT = pd.DataFrame(verdict_rows)
display(VERDICT)
print("NOTA sobre as linhas por classe: a 'réplica por classe' herda os acumuladores não repostos do código dos autores (a classe 3 é avaliada com contagens das classes 1-2 incluídas) e dá 0 às"
      " imagens onde a classe só aparece na predição. Numa execução real o padrão do artigo (granulação ~94) ficou muito mais próximo do Dice agregado padrão do que desta réplica, por isso"
      " o agregado padrão é a comparação por classe mais plausível; a réplica por classe fica só por transparência.")
print("Leitura: 'ponto_acima' compara só a estimativa pontual; só se pode falar de melhoria sólida se o IC exclui o valor do artigo (e mesmo assim"
      " a comparação depende da equivalência do protocolo e de um teste de 16 imagens, possivelmente com várias imagens do mesmo doente).")

# Comparações pareadas no TESTE (mesmas 16 imagens): quanto cada passo do pipeline acrescenta
labels = list(TEST_ROWS)
tp = [(f"TESTE: {labels[i][:40]}… vs {labels[0][:34]}…", TEST_ROWS[labels[i]]["res"], TEST_ROWS[labels[0]]["res"]) for i in range(1, len(labels))]
if len(labels) > 2:
    tp.append((f"TESTE: FINAL vs {labels[1][:34]}…", TEST_ROWS[labels[-1]]["res"], TEST_ROWS[labels[1]]["res"]))
TEST_PAIRED = paired_table(tp, ["macro_fg_dice", "paper_dice", "dice_fibrin", "dice_callus"])
display(TEST_PAIRED)
TEST_PAIRED.to_csv(DIRS["tables"] / "paired_test.csv", index=False)

# %% [markdown]
# ## 21b. Resumo no formato curto (tabelas por tecido e "Obtido vs Artigo")
# Mesmas métricas do artigo (IoU, Precision, Recall, DSC), em %, com a diferença em pontos percentuais. **Por tecido**: valores agregados sobre as 16 imagens de teste (definição padrão).
# **Global**: réplica do protocolo dos autores (`paper_*`), a única candidata a comparação direta (ver secção 12). "Superou" compara só as estimativas pontuais: com 16 imagens de teste
# o intervalo de confiança é largo, por isso a mensagem final indica sempre o IC do DSC.

# %%
TISSUES_PT = [("fibrin", "Fibrin"), ("granulation", "Granulation"), ("callus", "Callus")]


def tissue_table(m: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Tecido": nm, "IoU (%)": 100 * m[f"iou_{k}"], "Precision (%)": 100 * m[f"precision_{k}"], "Recall (%)": 100 * m[f"recall_{k}"], "DSC (%)": 100 * m[f"dice_{k}"]}
                         for k, nm in TISSUES_PT])


def vs_paper_table(m: dict, ref: dict) -> pd.DataFrame:
    rows = []
    for nm, key, rk in [("IoU", "paper_iou", "IoU"), ("Precision", "paper_precision", "Precision"), ("Recall", "paper_recall", "Recall"), ("DSC", "paper_dice", "Dice")]:
        got = 100 * m[key]
        rows.append({"Métrica": nm, "Obtido (%)": got, "Artigo (%)": ref[rk], "Diferença (p.p.)": got - ref[rk], "Superou": bool(got > ref[rk])})
    return pd.DataFrame(rows)


def pt(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def show(df: pd.DataFrame) -> None:
    with pd.option_context("display.float_format", "{:.3f}".format):
        display(df)


def ref_line() -> str:
    pc = PAPER["per_class_ssl"]
    return "Referência SSL do artigo — DSC:\n" + " | ".join(f"{nm} {pt(pc[nm]['Dice'])}%" for _, nm in TISSUES_PT)


SUMMARY: dict[str, pd.DataFrame] = {}
_labels = list(TEST_ROWS)
_roles = [(_labels[0], "Supervisionado"), (_labels[-1], "Final")] if len(_labels) < 3 else [(_labels[0], "Supervisionado"), (_labels[1], "SSL"), (_labels[-1], "Final")]
for label, role in _roles:
    m = TEST_ROWS[label]["metrics"]
    print(f"\n{role} — resultados por tecido  [{label}]")
    SUMMARY[f"tecido_{role}"] = tissue_table(m)
    show(SUMMARY[f"tecido_{role}"])
    print(ref_line())

print("\n" + "=" * 90)
for label, role, ref_name, ref in [(_labels[0], "Supervisionado", "artigo (SL)", PAPER["supervised"]), (_labels[-1], "Final", "SSL do artigo", PAPER["semi_supervised"])]:
    m, ci = TEST_ROWS[label]["metrics"], TEST_ROWS[label]["ci"]
    print(f"\n{role} vs {ref_name}")
    SUMMARY[f"vs_{role}"] = vs_paper_table(m, ref)
    show(SUMMARY[f"vs_{role}"])
    d = float(SUMMARY[f"vs_{role}"].loc[3, "Diferença (p.p.)"])
    lo, hi = 100 * ci["paper_dice"][0], 100 * ci["paper_dice"][1]
    if d < 0:
        print(f"⚠️ O DSC ficou {pt(abs(d))} p.p. abaixo do {ref_name}. (IC 95 % do nosso DSC: [{pt(lo)}–{pt(hi)}]%; {'o IC contém o valor do artigo' if lo <= ref['Dice'] <= hi else 'o IC não contém o valor do artigo'}.)")
    else:
        print(f"✅ O DSC superou o {ref_name} em {pt(d)} p.p. (IC 95 % do nosso DSC: [{pt(lo)}–{pt(hi)}]%; {'o IC contém o valor do artigo: diferença não significativa' if lo <= ref['Dice'] <= hi else 'o IC exclui o valor do artigo'}.)")
for k, df in SUMMARY.items():
    df.to_csv(DIRS["tables"] / f"summary_{k.replace(' ', '_')}.csv", index=False)
print("\nNota: o 'Global' usa a réplica do protocolo dos autores; a equivalência com a tabela do artigo é uma hipótese e o teste (16 imagens) é muito ruidoso.")
