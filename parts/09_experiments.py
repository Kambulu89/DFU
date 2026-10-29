# %% [markdown]
# ## 14. Treino supervisionado — sondas e validação cruzada (Experiências A e B)
#
# **Sondas (só validação oficial, só em `full`).** Duas sondas curtas com a arquitetura de referência e uma só semente: (i) *frame*/resolução
# (`padded 256` = réplica do artigo vs `roi 256/384/512`) e (ii) função de perda. Só se abandona a configuração por omissão se a alternativa ganhar por
# **mais de 0,005** de macro-Dice dos tecidos (histerese contra o ruído de 16 imagens).
#
# **Treino principal.**
# * `do_cv=True` (modo `full`): **validação cruzada estratificada de `cv_folds` folds** sobre as 94 imagens de desenvolvimento (treino+validação oficiais; **o teste nunca entra**).
#   Cada modelo treina em ~75 imagens e prevê as ~19 retidas; juntando os folds obtém-se uma previsão **fora de amostra (OOF) para as 94 imagens**, bem menos ruidosa que as 16 da
#   validação oficial. As épocas são fixas por arquitetura (`cv_epochs`) e o modelo é o **EMA final, sem seleção de época**, para o OOF não ter viés de seleção. O *ensemble* final usa os
#   modelos de todos os folds (portanto vê praticamente todos os dados).
# * `do_cv=False` (`quick`/`smoke`): divisão oficial 78/16/16, com seleção por *early stopping* na validação, como antes.
#
# Cada execução é idempotente (reutiliza `result.json`), por isso a sessão pode ser interrompida e retomada.

# %%
ALL_RESULTS: list[dict] = []
REF_ARCH = CFG["archs"][0]
PROBE_MARGIN = 0.005


def val_score(r: dict) -> float:
    v = r["val"]["macro_fg_dice"]
    return -1.0 if v is None or np.isnan(v) else v


def pick(results: list[dict], key_fn, default_key):
    """Escolhe a melhor configuração, mas só abandona a por omissão se ganhar por > PROBE_MARGIN."""
    by = {key_fn(r): r for r in results}
    best_key = max(by, key=lambda k: val_score(by[k]))
    if default_key in by and val_score(by[best_key]) - val_score(by[default_key]) <= PROBE_MARGIN:
        return default_key, by
    return best_key, by


if CFG["do_probes"]:
    # --- sonda 1: frame/resolução
    res = []
    for fr, sz in CFG["probe_frames"]:
        if not BUDGET.ok(600):
            break
        res.append(train_run(default_spec(exp="probe_frame", arch=REF_ARCH, frame=fr, size=sz, epochs=CFG["probe_epochs"], patience=CFG["probe_epochs"], keep_ckpt=False)))
    ALL_RESULTS += res
    key, by = pick(res, lambda r: (r["spec"]["frame"], r["spec"]["size"]), (CFG["frame"], CFG["size"]))
    print("Sonda frame/resolução (val macro-fg Dice):", {k: round(val_score(v), 4) for k, v in by.items()}, "->", key)
    CFG["frame"], CFG["size"] = key
    # --- sonda 2: perda
    res = []
    for ls in CFG["probe_losses"]:
        if not BUDGET.ok(600):
            break
        res.append(train_run(default_spec(exp="probe_loss", arch=REF_ARCH, loss=ls, epochs=CFG["probe_epochs"], patience=CFG["probe_epochs"], keep_ckpt=False)))
    ALL_RESULTS += res
    key, by = pick(res, lambda r: r["spec"]["loss"], CFG["loss"])
    print("Sonda de perda (val macro-fg Dice):", {k: round(val_score(v), 4) for k, v in by.items()}, "->", key)
    CFG["loss"] = key
else:
    print(f"Sondas desativadas: frame={CFG['frame']}, size={CFG['size']}, loss={CFG['loss']} (valores por omissão, NÃO validados neste notebook)")

# %%
SUP: list[dict] = []
if CFG["do_cv"]:
    # assert: o teste não entra nos folds
    _dev_names = {r["name"] for r in get_frame("roi")["train"] + get_frame("roi")["val"]}
    assert not (_dev_names & set(SPLIT_NAMES["test"])), "Fuga: imagens de teste no conjunto de desenvolvimento"
    _fm = fold_assignment(CFG["cv_folds"], CFG["cv_seed"])
    print("Folds (nº de imagens / com fibrina):", {f: (sum(v == f for v in _fm.values()),
          sum(1 for r in get_frame("roi")["train"] + get_frame("roi")["val"] if _fm[r["name"]] == f and (r["mask"] == 1).any())) for f in range(CFG["cv_folds"])})
    # arquitetura como ciclo exterior: se o tempo acabar, as primeiras arquiteturas (as de maior prioridade) ficam com todos os folds
    for seed in CFG["seeds"]:
        for arch in CFG["archs"]:
            for fold in range(CFG["cv_folds"]):
                ep = CFG["cv_epochs"].get(arch, CFG["epochs"])
                old = next((r for r in REUSED_RUNS if r["spec"]["exp"] == "cv" and reusable(r, arch=arch, seed=seed, fold=fold, n_folds=CFG["cv_folds"], epochs=ep)), None)
                if old is not None:
                    print(f"[reaproveitado] {arch} seed {seed} fold {fold}: {old['run_id']}")
                    SUP.append(old)
                    continue
                if not BUDGET.ok(900):
                    print(f"Orçamento de tempo quase esgotado: a parar (último: {arch} fold {fold}).")
                    break
                SUP.append(train_run(default_spec(exp="cv", arch=arch, seed=seed, fold=fold, n_folds=CFG["cv_folds"], selection="last", epochs=ep, patience=ep)))
else:
    for seed in CFG["seeds"]:           # sementes no ciclo exterior: se o tempo acabar, todas as arquiteturas têm pelo menos a 1.ª semente
        for arch in CFG["archs"]:
            if not BUDGET.ok(900):
                print("Orçamento de tempo quase esgotado: a parar novas execuções supervisionadas.")
                break
            SUP.append(train_run(default_spec(exp="sup", arch=arch, seed=seed)))
ALL_RESULTS += SUP
assert SUP, "Nenhuma execução supervisionada concluída"


def runs_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        v, s = r["val"], r["spec"]
        rows.append(dict(exp=s["exp"], arch=s["arch"], seed=s["seed"], fold=s.get("fold"), frame=s["frame"], size=s["size"], loss=s["loss"], pretrained=r["provenance"]["pretrained_effective"],
                         best_ep=r["best_epoch"], epochs_run=r["epochs_run"], status=r["status"], macro_fg_dice=v["macro_fg_dice"], micro_fg_dice=v["micro_fg_dice"],
                         paper_dice=v["paper_dice"], dice_fibrin=v["dice_fibrin"], dice_granulation=v["dice_granulation"], dice_callus=v["dice_callus"],
                         macro_fg_iou=v["macro_fg_iou"], minutes=round(r["train_seconds"] / 60, 1)))
    return pd.DataFrame(rows)


SUP_TABLE = runs_table(SUP)
SUP_TABLE.to_csv(DIRS["tables"] / "supervised_runs_val.csv", index=False)
print("Validação por execução (CV: fold retido; oficial: val de 16 imagens) — NÃO é o teste:")
display(SUP_TABLE.round(4))
agg = SUP_TABLE.groupby("arch")[["macro_fg_dice", "micro_fg_dice", "paper_dice", "dice_fibrin", "dice_granulation", "dice_callus"]].agg(["mean", "std"]).round(4)
display(agg)
agg.to_csv(DIRS["tables"] / "supervised_by_arch_val.csv")
if not SUP_TABLE.pretrained.all():
    print("AVISO: há execuções SEM pesos pré-treinados (ver weights_provenance.json). Não são comparáveis a transfer learning.")

# %%
# Curvas de aprendizagem (loss treino/val, Dice, IoU) para a melhor execução de cada arquitetura
best_by_arch = {}
for r in SUP:
    if r["spec"]["arch"] not in best_by_arch or val_score(r) > val_score(best_by_arch[r["spec"]["arch"]]):
        best_by_arch[r["spec"]["arch"]] = r
fig, axes = plt.subplots(len(best_by_arch), 3, figsize=(16, 3.4 * len(best_by_arch)), squeeze=False)
for row, (arch, r) in zip(axes, best_by_arch.items()):
    h = pd.DataFrame(r["history"])
    row[0].plot(h.epoch, h.train_loss, label="treino"); row[0].plot(h.epoch, h.val_loss, label="validação (EMA)"); row[0].set_title(f"{arch} (fold {r['spec'].get('fold')}) — loss"); row[0].legend()
    for c in ["macro_fg_dice", "dice_fibrin", "dice_granulation", "dice_callus"]:
        row[1].plot(h.epoch, h[f"val_{c}"], label=c.replace("val_", ""))
    row[1].set_title("Dice de validação"); row[1].legend(fontsize=7); row[1].set_ylim(0, 1)
    row[2].plot(h.epoch, h.val_macro_fg_iou); row[2].set_title("macro-IoU (tecidos) de validação"); row[2].set_ylim(0, 1)
    for a in row:
        a.axvline(r["best_epoch"], color="k", ls=":", lw=0.8); a.set_xlabel("época")
fig.tight_layout(); savefig(fig, "learning_curves_supervised.png"); plt.show()
