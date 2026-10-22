# %% [markdown]
# ## 20. Visualização das previsões no teste
# Imagem, GT, previsão e *overlay* transparente; melhores e piores casos segundo o Dice médio dos tecidos por imagem (GT∪predição).

# %%
FINAL_ROW = TEST_ROWS[FINAL_LABEL]
FINAL_PREDS = probs_to_preds(FINAL_ROW["probs"], FINAL_ROW["pp"])


def per_image_table(res: EvalResult, items: EvalItems) -> pd.DataFrame:
    rows = []
    for i, cm in enumerate(res.cms):
        pc = per_class_from_cm(cm)
        d = pc["dice"][TISSUES]
        rows.append(dict(name=items.names[i], h=items.geoms[i]["h"], w=items.geoms[i]["w"], fg_frac=cm[1:].sum() / cm.sum(),
                         dice_fibrin=d[0], dice_granulation=d[1], dice_callus=d[2], mean_dice=np.nanmean(d) if (~np.isnan(d)).any() else np.nan,
                         gt_fibrin_px=int(cm[1].sum()), pred_fibrin_px=int(cm[:, 1].sum())))
    return pd.DataFrame(rows)


PER_IMG = per_image_table(FINAL_ROW["res"], TEST_ITEMS)
PER_IMG.to_csv(DIRS["tables"] / "test_per_image_final.csv", index=False)


def show_predictions(items: EvalItems, preds: list[np.ndarray], idxs: list[int], title: str, fname: str, ncols: int = 4) -> None:
    n = len(idxs)
    fig, axes = plt.subplots(n, ncols, figsize=(3.1 * ncols, 3.1 * n), squeeze=False)
    for r, i in enumerate(idxs):
        img, gt, pr = items.imgs[i], items.gts[i], preds[i]
        for a, im, t in zip(axes[r], [img, colorize(gt), colorize(pr), overlay(img, pr)], ["imagem", "GT", "previsão", "overlay (previsão)"]):
            a.imshow(im); a.axis("off"); a.set_title(f"{items.names[i]} — {t}" if t == "imagem" else t, fontsize=8)
        d = PER_IMG.iloc[i]
        axes[r, 2].set_title(f"previsão (Dice méd. {d.mean_dice:.2f})", fontsize=8)
    fig.suptitle(title); fig.tight_layout(); savefig(fig, fname); plt.show()


order = PER_IMG.mean_dice.fillna(-1).sort_values().index.tolist()
show_predictions(TEST_ITEMS, FINAL_PREDS, order[-4:][::-1], "Teste — melhores 4 casos (modelo final)", "test_best_cases.png")
show_predictions(TEST_ITEMS, FINAL_PREDS, order[:4], "Teste — piores 4 casos (modelo final)", "test_worst_cases.png")

# %%
# Matriz de confusão por píxel (agregada no teste, linhas normalizadas)
cm_sum = FINAL_ROW["res"].cms.sum(0)
cm_norm = cm_sum / cm_sum.sum(1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, mat, ttl, f in [(ax[0], cm_norm, "normalizada por linha (recall)", ".2f"), (ax[1], np.log10(cm_sum + 1), "log10(contagem+1)", ".1f")]:
    im = a.imshow(mat, cmap="Blues"); a.set_xticks(range(K)); a.set_yticks(range(K)); a.set_xticklabels(CLASS_NAMES, rotation=30); a.set_yticklabels(CLASS_NAMES)
    a.set_xlabel("previsto"); a.set_ylabel("GT"); a.set_title(f"Matriz de confusão (teste) — {ttl}")
    for i in range(K):
        for j in range(K):
            a.text(j, i, format(mat[i, j], f), ha="center", va="center", color="white" if mat[i, j] > mat.max() * 0.6 else "black", fontsize=8)
fig.tight_layout(); savefig(fig, "test_confusion_matrix.png"); plt.show()

# Gráficos comparativos de modelos (teste) com IC
fig, ax = plt.subplots(figsize=(9, 3.6))
labels = list(TEST_ROWS)
for off, key in enumerate(["dice_fibrin", "dice_granulation", "dice_callus"]):
    vals = [TEST_ROWS[l]["metrics"][key] for l in labels]
    err = np.array([[TEST_ROWS[l]["metrics"][key] - TEST_ROWS[l]["ci"][key][0], TEST_ROWS[l]["ci"][key][1] - TEST_ROWS[l]["metrics"][key]] for l in labels]).T
    ax.bar(np.arange(len(labels)) + 0.27 * (off - 1), vals, 0.27, yerr=err, label=key.replace("dice_", ""), capsize=2)
ax.set_xticks(range(len(labels))); ax.set_xticklabels([l.split(":")[1].split("(")[0].strip()[:22] for l in labels], fontsize=7); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Dice por classe no teste (agregado; IC bootstrap 95 %)")
fig.tight_layout(); savefig(fig, "test_model_comparison.png"); plt.show()

# %% [markdown]
# ## 22. Análise de erros e limitações
# O texto abaixo é **gerado a partir dos números desta execução** (nenhuma frase foi escrita de antemão sobre resultados).

# %%
def boundary_error_share(gts, preds, band: int = 3) -> float:
    """Fração dos píxeis errados (excluindo erros só de fundo<->fundo) que estão a <= `band` px de uma fronteira entre classes no GT."""
    err_tot = err_near = 0
    for g, p in zip(gts, preds):
        edge = np.zeros(g.shape, np.uint8)
        for c in range(K):
            m = (g == c).astype(np.uint8)
            edge |= (cv2.dilate(m, np.ones((3, 3), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8)))
        near = cv2.dilate(edge, np.ones((2 * band + 1, 2 * band + 1), np.uint8)).astype(bool)
        err = g != p
        err_tot += int(err.sum()); err_near += int((err & near).sum())
    return err_near / max(err_tot, 1)


off = cm_sum.astype(float).copy(); np.fill_diagonal(off, 0)
conf_pairs = sorted(((off[i, j] / cm_sum[i].sum(), CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(1, K) for j in range(K) if i != j), reverse=True)[:4]
bshare = boundary_error_share(TEST_ITEMS.gts, FINAL_PREDS)
size_corr = PER_IMG[["h", "mean_dice"]].dropna().corr(method="spearman").iloc[0, 1] if PER_IMG.mean_dice.notna().sum() > 2 else float("nan")
fib_img = PER_IMG[PER_IMG.gt_fibrin_px > 0]
print("Principais confusões (fração dos píxeis de GT da classe, linha->coluna):")
for v, a, b in conf_pairs:
    print(f"  {a:12s} -> {b:12s}: {100 * v:5.1f} %")
print(f"Fração dos píxeis errados a <=3 px de uma fronteira do GT: {100 * bshare:.1f} %  (alta => erros dominados por ambiguidade de fronteira)")
print(f"Correlação de Spearman entre tamanho da imagem original e Dice médio por imagem: {size_corr:.2f}")
print(f"Fibrina: {len(fib_img)}/{len(PER_IMG)} imagens de teste têm fibrina no GT; Dice mediano por imagem nessas: {fib_img.dice_fibrin.median():.2f}; "
      f"piores: {', '.join(fib_img.sort_values('dice_fibrin').name.head(3))}")
print(f"Fibrina prevista vs GT (píxeis, teste): {int(PER_IMG.pred_fibrin_px.sum())} vs {int(PER_IMG.gt_fibrin_px.sum())} "
      f"-> {'sobre-segmentação' if PER_IMG.pred_fibrin_px.sum() > 1.15 * PER_IMG.gt_fibrin_px.sum() else ('sub-segmentação' if PER_IMG.pred_fibrin_px.sum() < 0.85 * PER_IMG.gt_fibrin_px.sum() else 'volume comparável')}")
display(PER_IMG.round(3).sort_values("mean_dice").head(8))

# %% [markdown]
# **Limitações (válidas independentemente dos números):**
# 1. **Teste de 16 imagens**: os IC *bootstrap* são largos; diferenças de 1–3 pontos de Dice não são estatisticamente distinguíveis. Imagens do mesmo doente podem
#    estar em divisões diferentes (não há identificador de doente) — a generalização pode estar sobrestimada, também no artigo.
# 2. **Equivalência de protocolo não garantida**: o `paper_*` replica o código dos autores, mas não temos confirmação de que as tabelas do artigo foram geradas exatamente
#    assim, nem dos checkpoints/sementes deles. Em particular, o acumulador não reposto torna esse "global" diferente de um micro-Dice padrão.
# 3. **Resolução**: as imagens originais são pequenas (67–266 px); ampliar para 256/384 não recupera detalhe, apenas dá mais *tokens* úteis ao encoder.
# 4. **Seleção em 16 imagens de validação** (mitigada com histerese, sementes múltiplas e ensemble uniforme) pode ainda sobreajustar a validação.
# 5. **SSL**: não há GT para os não anotados; a qualidade dos *pseudo-labels* só é monitorizada indiretamente. O ganho tem de ser lido contra o ponto inicial supervisionado.
# 6. **Pesos pré-treinados**: se a execução registou `random_init` em `weights_provenance.json`, os resultados **não** representam *transfer learning*.
#
# **Próxima experiência mais promissora** (depende dos resultados acima): validação cruzada estratificada *k*-fold sobre as 94 imagens de desenvolvimento (sem tocar no teste)
# para selecionar hiperparâmetros com menos ruído; resolução 384 se a sonda a favoreceu; mais sementes no *ensemble*; ou *self-supervised pre-training* nos 600 não anotados se
# o SSL se mostrar limitado pela qualidade das *features*.

# %% [markdown]
# ## 23. Inferência sobre uma imagem nova
# `segment_image` aceita um caminho ou um *array* RGB, faz *letterbox*, TTA, *ensemble* do candidato final, reverte para o tamanho original e aplica o pós-processamento
# escolhido na validação. Para fotografias novas, `assume_zero_padded=False` (a imagem inteira é o conteúdo). As imagens do dataset estão coladas num canto preto, por
# isso a demonstração usa `assume_zero_padded=True` (recorta a extensão do conteúdo).

# %%
def segment_image(image, ids: list[str] | None = None, tta: str | None = None, pp: float | None = None, assume_zero_padded: bool = False, return_probs: bool = False):
    ids, tta, pp = ids or FINAL["ids"], tta or FINAL["tta"], FINAL["pp"] if pp is None else pp
    img = read_rgb(Path(image)) if isinstance(image, (str, Path)) else np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3].astype(np.uint8)
    if assume_zero_padded:
        h, w = content_extent(img)
        img = np.ascontiguousarray(img[:h, :w])
    items = EvalItems([dict(name="new", img=img, mask=None)], SIZE, labeled=False)
    probs = predict_probs([load_pool_model(i) for i in ids], items, tta)[0]
    pred = remove_small_components(probs.argmax(0).astype(np.uint8), pp)
    return (img, pred, probs) if return_probs else (img, pred)


demo = []
if len(UNL):
    demo.append(("não anotada (dataset)", UNL[0]["path"], True))
demo.append(("teste (apenas visualização)", _locate("Padded", "Images", SPLIT_NAMES["test"][0]), True))
fig, axes = plt.subplots(len(demo), 3, figsize=(10, 3.4 * len(demo)), squeeze=False)
for r, (nm, path, zp) in enumerate(demo):
    img, pred = segment_image(path, assume_zero_padded=zp)
    for a, im, t in zip(axes[r], [img, colorize(pred), overlay(img, pred)], [f"{nm}", "segmentação", "overlay"]):
        a.imshow(im); a.axis("off"); a.set_title(t, fontsize=9)
fig.tight_layout(); savefig(fig, "inference_demo.png"); plt.show()
print("Fração de píxeis por classe na imagem demo:", dict(zip(CLASS_NAMES, np.round(np.bincount(pred.ravel(), minlength=K) / pred.size, 3))))

# %% [markdown]
# ## 24. Exportação de resultados e modelos
# Em `OUT`: `config.json` (configuração, sementes, versões, hardware), `weights_provenance.json`, `tables/*.csv|json`, `figures/*.png`, `runs/*/result.json` (histórico de treino),
# `models/*.pt` (estados em fp16 do candidato final) e `final_bundle.json` (como reconstruir o *ensemble*). Em Kaggle ficam em `/kaggle/working`.

# %%
(OUT / "config.json").write_text(json.dumps(dict(cfg=CFG, hw=HW, seeds=CFG["seeds"], mode=MODE, data_root=str(DATA_ROOT), split_sizes={k: len(v) for k, v in SPLIT_NAMES.items()},
                                                 n_unlabeled_used=len(UNL), n_unlabeled_excluded_dups=len(EXCLUDED_UNL)), indent=1, default=str))
(OUT / "weights_provenance.json").write_text(json.dumps(WEIGHTS_PROVENANCE, indent=1, default=str))
(OUT / "test_ledger.json").write_text(json.dumps(TEST_LEDGER, indent=1))
final_members = []
for rid in FINAL["ids"]:
    dst = DIRS["models"] / f"{rid}.pt"
    shutil.copy(POOL[rid]["ckpt"], dst)
    final_members.append(dict(file=f"models/{dst.name}", arch=POOL[rid]["arch"], kind=POOL[rid]["kind"], seed=POOL[rid]["seed"], val_macro_fg_dice=val_score(POOL[rid]["run"])))
(OUT / "final_bundle.json").write_text(json.dumps(dict(
    candidate=FINAL_NAME, members=final_members, frame=FRAME, size=SIZE, tta=FINAL["tta"], pp_min_area_frac=FINAL["pp"], classes=CLASS_NAMES, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    note="Modelos = smp.<arch>(encoder_weights=None, classes=4) envolvidos em Net (normalização ImageNet interna); state_dict em fp16 -> converter para float32 ao carregar.",
    test_metrics={k: v for k, v in FINAL_ROW["metrics"].items()}, test_ci_95={k: list(v) for k, v in FINAL_ROW["ci"].items()}), indent=1, default=float))
pd.DataFrame([dict(label=l, **r["metrics"]) for l, r in TEST_ROWS.items()]).to_csv(DIRS["tables"] / "test_metrics_all_definitions.csv", index=False)
all_hist = [dict(run_id=r["run_id"], **h) for r in ALL_RESULTS for h in r["history"]]
pd.DataFrame(all_hist).to_csv(DIRS["tables"] / "training_history_all_runs.csv", index=False)
runs_table([r for r in ALL_RESULTS if "val" in r]).to_csv(DIRS["tables"] / "all_runs_val.csv", index=False)
(DIRS["tables"] / "paper_reported_constants.json").write_text(json.dumps(PAPER, indent=1))

def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:  # `tabulate` ausente: texto simples
        return "```\n" + df.to_string(index=False) + "\n```"


report = [f"# Relatório de experiências — modo `{MODE}`", "", f"* Hardware: {HW['gpu_name'] or 'CPU'} ({HW['vram_gb']} GB) | torch {HW['torch']} | smp {HW['smp']} | timm {HW['timm']}",
          f"* Frame/resolução: `{FRAME}`/{SIZE} | perda: `{CFG['loss']}` | sementes: {CFG['seeds']} | arquiteturas: {CFG['archs']}",
          f"* Pesos pré-treinados efetivos: {{{', '.join(f'{k}: {v.get('pretrained_effective')}' for k, v in WEIGHTS_PROVENANCE.items())}}}",
          f"* Não anotadas usadas: {len(UNL)} (excluídas por quase-duplicado de val/teste: {len(EXCLUDED_UNL)})", f"* Candidato final (escolhido na validação): `{FINAL_NAME}` — {len(FINAL['ids'])} modelo(s), TTA={FINAL['tta']}, pp={FINAL['pp']}",
          f"* Eventos de leitura do teste: {len(TEST_LEDGER)} (ver test_ledger.json)", "", "## Tabela comparativa (teste; % ; IC bootstrap 95 %)", "", md_table(COMPARISON),
          "", "## Veredicto calculado", "", VERDICT.to_string(index=False), "",
          "Aviso: os valores acima só têm valor científico se `weights_provenance.json` indicar pesos pré-treinados e se o modo não for `smoke`."]
(OUT / "experiments_report.md").write_text("\n".join(report))
print("Exportado em", OUT)
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.suffix in (".json", ".csv", ".md", ".png") and "runs" not in p.parts:
        print("  ", p.relative_to(OUT), f"{p.stat().st_size / 1024:.0f} KB")
print("\nMODO:", MODE, "| " + ("ATENÇÃO: modo smoke = teste de pipeline; estes números NÃO têm valor científico." if MODE == "smoke" else "execução científica."))
