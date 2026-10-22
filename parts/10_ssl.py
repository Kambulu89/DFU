# %% [markdown]
# ## 16. Aprendizagem semissupervisionada (Experiência C)
#
# Em vez da escolha aleatória de *pseudo-labels* do artigo, usa-se um **EMA *teacher–student* online** (Mean Teacher + FixMatch para segmentação),
# partindo do **melhor checkpoint supervisionado** (escolhido na validação):
#
# 1. o *teacher* (EMA do *student*) prevê, com TTA de flips, a vista **fraca** de cada imagem **sem anotação** → probabilidade e rótulo por píxel;
# 2. **limiares por classe adaptativos**: para cada classe mantém-se a fração `ssl_keep_frac` de píxeis mais confiantes (percentil com média móvel),
#    limitada a `[ssl_tau_min, ssl_tau_max]`; píxeis abaixo do limiar são **ignorados** (`ignore_index`) — evita que a fibrina (minoritária) seja
#    engolida e reduz o viés de confirmação;
# 3. o *student* recebe a vista **forte** (fotometria forte + **CutMix** entre imagens, misturando também *pseudo-labels* e máscaras de confiança) e aprende com
#    a perda de consistência, ponderada por um *ramp-up* sigmoidal, **somada** à perda supervisionada nas imagens anotadas (âncora anti-deriva);
# 4. avalia-se o *teacher* na validação a cada "época"; guarda-se o melhor (inclui o ponto inicial: se o SSL não melhorar, **mantém-se o supervisionado**);
# 5. **rondas iterativas**: o melhor da ronda inicializa a seguinte; se uma ronda não melhorar a validação, o processo pára e faz *rollback*.
#
# Anti-fuga: os não anotados quase-duplicados de val/teste (dHash) foram removidos na auditoria; **nenhuma imagem/máscara de teste é lida aqui**.
# Os *pseudo-labels* não têm GT, por isso a sua qualidade só é monitorizada por confiança e fração retida por classe (gráficos abaixo).

# %%
class ClassThresholds:
    """Limiar de confiança por classe: percentil (1-keep_frac) das confianças dos píxeis previstos como essa classe, com média móvel e limites."""

    def __init__(self, tau_min: float, tau_max: float, keep_frac: float, momentum: float = 0.9):
        self.tau = np.full(K, (tau_min + tau_max) / 2)
        self.lo, self.hi, self.keep, self.m = tau_min, tau_max, keep_frac, momentum

    def update(self, conf: torch.Tensor, pl: torch.Tensor) -> None:
        for c in range(K):
            sel = conf[pl == c]
            if sel.numel() >= 200:
                q = float(torch.quantile(sel.float()[:2_000_000], 1 - self.keep))
                self.tau[c] = self.m * self.tau[c] + (1 - self.m) * float(np.clip(q, self.lo, self.hi))

    def as_tensor(self, device) -> torch.Tensor:
        return torch.tensor(self.tau, dtype=torch.float32, device=device)


def strong_photometric(x: torch.Tensor) -> torch.Tensor:
    """Fotometria forte em GPU (x em [0,1], (B,3,H,W)); o padding preto mantém-se preto."""
    B, dev = x.shape[0], x.device
    valid = (x.amax(1, keepdim=True) > 0).float()
    r = lambda lo, hi: torch.empty(B, 1, 1, 1, device=dev).uniform_(lo, hi)  # noqa: E731
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * r(0.7, 1.3) + m + r(-0.15, 0.15)
    g = x.mean(1, keepdim=True)
    x = g + (x - g) * r(0.7, 1.3)
    x = x.clamp(1e-4, 1) ** r(0.7, 1.4)
    k = torch.tensor([1, 2, 1], device=dev, dtype=x.dtype)
    k = (k[:, None] * k[None]) / 16.0
    blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k.expand(3, 1, 3, 3), groups=3)
    x = torch.where(torch.rand(B, 1, 1, 1, device=dev) < 0.3, blur, x)
    x = x + torch.randn_like(x) * r(0.0, 0.04) * (torch.rand(B, 1, 1, 1, device=dev) < 0.5)
    return x.clamp(0, 1) * valid


def cutmix(x: torch.Tensor, pl: torch.Tensor, keep: torch.Tensor):
    """CutMix: cola uma caixa de outra imagem do batch; mistura imagem, pseudo-label e máscara de confiança de forma consistente."""
    B, _, H, W = x.shape
    perm = torch.randperm(B, device=x.device)
    M = torch.zeros(B, 1, H, W, dtype=torch.bool, device=x.device)
    for i in range(B):
        area = random.uniform(0.15, 0.4) * H * W
        ar = random.uniform(0.5, 2.0)
        h, w = int(min(H, math.sqrt(area * ar))), int(min(W, math.sqrt(area / ar)))
        y0, x0 = random.randint(0, H - h), random.randint(0, W - w)
        M[i, :, y0:y0 + h, x0:x0 + w] = True
    return torch.where(M, x[perm], x), torch.where(M[:, 0], pl[perm], pl), torch.where(M[:, 0], keep[perm], keep)


def rampup(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return math.exp(-5.0 * (1.0 - t) ** 2)


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def train_ssl_round(sup_run: dict, init_ckpt: str, round_idx: int, unl_recs: list[dict]) -> dict:
    """Uma ronda de EMA teacher–student a partir de `init_ckpt`. Devolve resultado com o melhor teacher (inclui o ponto inicial)."""
    spec = dict(sup_run["spec"]) | dict(exp=f"ssl_r{round_idx}", epochs=CFG["ssl_epochs"], init=init_ckpt, keep_frac=CFG["ssl_keep_frac"], tau=[CFG["ssl_tau_min"], CFG["ssl_tau_max"]],
                lambda_u=CFG["ssl_lambda_u"], ssl_steps=CFG["ssl_steps_per_epoch"], n_unl=len(unl_recs))
    rid = run_id_of(spec)
    rdir = DIRS["runs"] / rid
    rf = rdir / "result.json"
    if rf.exists() and CFG["resume"]:
        print(f"[reutilizado] {rid}")
        return json.loads(rf.read_text())
    rdir.mkdir(parents=True, exist_ok=True)
    seed_everything(spec["seed"] + 1000 * (round_idx + 1))
    S, bs = spec["size"], spec["batch_size"]
    recs = get_frame(spec["frame"])
    lab_set, unl_set = SegSet(recs["train"], S, aug="standard"), SegSet(unl_recs, S, aug="weak", labeled=False)
    val_items = EvalItems(recs["val"], S)
    nw = CFG["num_workers"]
    mk = lambda ds, sampler: torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=nw, drop_last=True, worker_init_fn=worker_init,  # noqa: E731
                                                          pin_memory=DEVICE.type == "cuda", persistent_workers=nw > 0)
    big = 10**6
    lab_w = torch.as_tensor(lab_set.sample_weights(CFG["fibrin_boost"]), dtype=torch.double)
    lab_it = _cycle(mk(lab_set, torch.utils.data.WeightedRandomSampler(lab_w, big, replacement=True)))
    unl_it = _cycle(mk(unl_set, torch.utils.data.RandomSampler(unl_set, replacement=True, num_samples=big)))

    student, prov = Net(ARCH_SPECS[spec["arch"]]["ctor"](None)), sup_run["provenance"]   # pesos vêm do checkpoint supervisionado
    student.load_state_dict(_load_half(Path(init_ckpt)))
    student.to(DEVICE)
    teacher = EMA(student, 0.99)
    micro = max(1, probe_micro_batch(student, S, bs) // 2)
    crit = SegLoss(spec["loss"], class_weights_from(recs["train"], CFG["class_weight_cap"]), CFG["label_smoothing"]).to(DEVICE)
    lr_mult = 0.25
    opt = torch.optim.AdamW([{"params": student.encoder_params(), "lr": spec["lr_encoder"] * lr_mult}, {"params": student.other_params(), "lr": spec["lr_decoder"] * lr_mult}],
                            weight_decay=CFG["weight_decay"])
    total = spec["epochs"] * spec["ssl_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: lr_factor(st, int(0.05 * total), total, 0.05))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    thr = ClassThresholds(CFG["ssl_tau_min"], CFG["ssl_tau_max"], CFG["ssl_keep_frac"])

    base = evaluate_models([teacher.module], val_items, "none")[0].metrics()
    base_score = base["macro_fg_dice"]
    best, best_ep, bad = base_score, -1, 0
    shutil.copy(init_ckpt, rdir / "best.pt")
    hist, t0 = [], time.time()
    acc = dict(lsup=0.0, lunl=0.0, n=0, kept=np.zeros(K), seen=np.zeros(K), valid_frac=0.0)
    step = 0
    for epoch in range(spec["epochs"]):
        if not BUDGET.ok(60):
            break
        student.train()
        for _ in range(spec["ssl_steps"]):
            xl, yl = next(lab_it); xu = next(unl_it)
            xl, yl, xu = xl.to(DEVICE), yl.to(DEVICE), xu.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            for xc, yc in zip(xl.split(micro), yl.split(micro)):          # perda supervisionada (âncora)
                with AUTOCAST():
                    lg = student(xc)
                ls = crit(lg.float(), yc) * len(xc) / len(xl)
                scaler.scale(ls).backward(); acc["lsup"] += float(ls)
            with torch.no_grad():                                          # pseudo-labels do teacher (vista fraca, TTA de flips)
                p = torch.cat([canvas_probs(teacher.module, c, "flip") for c in xu.split(micro)])
                conf, pl = p.max(1)
            thr.update(conf, pl)
            keep = conf >= thr.as_tensor(DEVICE)[pl]
            for c in range(K):
                acc["seen"][c] += int((pl == c).sum()); acc["kept"][c] += int(((pl == c) & keep).sum())
            xs = strong_photometric(xu)
            xs, pl2, keep2 = cutmix(xs, pl, keep)
            tgt = torch.where(keep2, pl2, torch.full_like(pl2, IGNORE))
            w_u = spec["lambda_u"] * rampup(step / max(1.0, 0.3 * total))
            for xc, tc in zip(xs.split(micro), tgt.split(micro)):
                nvalid = int((tc != IGNORE).sum())
                if nvalid < 32:
                    continue
                with AUTOCAST():
                    lg = student(xc)
                lu = F.cross_entropy(lg.float(), tc, ignore_index=IGNORE) * (len(xc) / len(xs)) * w_u
                scaler.scale(lu).backward(); acc["lunl"] += float(lu)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), CFG["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step(); teacher.update(student)
            acc["n"] += 1; step += 1
        vm = evaluate_models([teacher.module], val_items, "none")[0].metrics()
        score = 0.0 if np.isnan(vm["macro_fg_dice"]) else vm["macro_fg_dice"]
        hist.append(dict(epoch=epoch, loss_sup=acc["lsup"] / acc["n"], loss_unl=acc["lunl"] / acc["n"], val_macro_fg_dice=vm["macro_fg_dice"], val_micro_fg_dice=vm["micro_fg_dice"],
                         val_dice_fibrin=vm["dice_fibrin"], val_dice_granulation=vm["dice_granulation"], val_dice_callus=vm["dice_callus"], val_macro_fg_iou=vm["macro_fg_iou"],
                         val_paper_dice=vm["paper_dice"], w_u=w_u, **{f"tau_{CLASS_NAMES[c].lower()}": float(thr.tau[c]) for c in range(K)},
                         **{f"kept_{CLASS_NAMES[c].lower()}": float(acc["kept"][c] / max(acc["seen"][c], 1)) for c in range(K)},
                         **{f"share_{CLASS_NAMES[c].lower()}": float(acc["seen"][c] / max(acc["seen"].sum(), 1)) for c in range(K)}))
        acc = dict(lsup=0.0, lunl=0.0, n=0, kept=np.zeros(K), seen=np.zeros(K), valid_frac=0.0)
        if score > best + 1e-4:
            best, best_ep, bad = score, epoch, 0
            _save_half(teacher.module.state_dict(), rdir / "best.pt")
        else:
            bad += 1
        if epoch % max(1, spec["epochs"] // 8) == 0 or epoch == spec["epochs"] - 1:
            print(f"  {rid} ep{epoch:03d} Lsup {hist[-1]['loss_sup']:.3f} Lunl {hist[-1]['loss_unl']:.3f} val {score:.4f} (base {base_score:.4f}, best {best:.4f}@{best_ep})")
        if bad >= max(8, spec["epochs"] // 3):
            break

    final = Net(ARCH_SPECS[spec["arch"]]["ctor"](None)).to(DEVICE)
    final.load_state_dict(_load_half(rdir / "best.pt"))
    vfin = evaluate_models([final], val_items, "none")[0].metrics()
    result = dict(spec=spec, run_id=rid, status="finished", best_epoch=best_ep, epochs_run=len(hist), val=vfin, val_before=base, improved=best > base_score + 1e-4, history=hist,
                  provenance=prov, params_M=round(count_params(student), 2), micro_batch=micro, train_seconds=round(time.time() - t0, 1), ckpt=str(rdir / "best.pt"),
                  device=HW["gpu_name"] or "cpu")
    rf.write_text(json.dumps(result, indent=1, default=float))
    del student, teacher, opt, final
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


SSL_RESULTS: list[dict] = []
SSL_FINAL: dict[str, dict] = {}   # id da execução supervisionada -> melhor resultado SSL (ou o supervisionado se não houve ganho)
if CFG["do_ssl"] and len(UNL) > 0 and BUDGET.ok(1800):
    arch_mean = SUP_TABLE.groupby("arch").macro_fg_dice.mean()
    ssl_arch = arch_mean.idxmax()
    cand = sorted([r for r in SUP if r["spec"]["arch"] == ssl_arch], key=val_score, reverse=True)
    targets = cand if CFG["ssl_all_seeds"] else cand[:1]
    unl_recs = load_unlabeled_records(CFG["frame"])
    print(f"SSL sobre {ssl_arch} | {len(unl_recs)} imagens sem anotação (após filtro anti-fuga) | alvos: {[t['run_id'] for t in targets]}")
    for tr in targets:
        cur, prev_score = tr, val_score(tr)
        for rnd in range(CFG["ssl_rounds"]):
            if not BUDGET.ok(1200):
                print("Orçamento de tempo: a parar rondas SSL."); break
            r = train_ssl_round(tr, cur["ckpt"], rnd, unl_recs)
            SSL_RESULTS.append(r)
            gain = val_score(r) - prev_score
            print(f"  ronda {rnd}: val macro-fg Dice {prev_score:.4f} -> {val_score(r):.4f} (Δ={gain:+.4f}) {'MELHOROU' if r['improved'] else 'sem ganho -> rollback e paragem'}")
            if not r["improved"]:
                break
            cur, prev_score = r, val_score(r)
        SSL_FINAL[tr["run_id"]] = cur
    ALL_RESULTS += SSL_RESULTS
elif not CFG["do_ssl"]:
    print("SSL desativado na configuração.")
else:
    print("SSL OMITIDO: sem imagens não anotadas utilizáveis (ou sem orçamento de tempo). O notebook continua em modo supervisionado.")

# %%
if SSL_RESULTS:
    rows = [dict(run=r["run_id"], ronda=r["spec"]["exp"], antes=r["val_before"]["macro_fg_dice"], depois=r["val"]["macro_fg_dice"],
                 fibrin_antes=r["val_before"]["dice_fibrin"], fibrin_depois=r["val"]["dice_fibrin"], callus_antes=r["val_before"]["dice_callus"], callus_depois=r["val"]["dice_callus"],
                 granulation_antes=r["val_before"]["dice_granulation"], granulation_depois=r["val"]["dice_granulation"], melhorou=r["improved"], melhor_epoca=r["best_epoch"]) for r in SSL_RESULTS]
    SSL_TABLE = pd.DataFrame(rows)
    SSL_TABLE.to_csv(DIRS["tables"] / "ssl_before_after_val.csv", index=False)
    print("Antes vs depois do SSL (VALIDAÇÃO):")
    display(SSL_TABLE.round(4))
    r = SSL_RESULTS[0]
    h = pd.DataFrame(r["history"])
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.4))
    ax[0].plot(h.epoch, h.loss_sup, label="supervisionada"); ax[0].plot(h.epoch, h.loss_unl, label="não supervisionada (ponderada)"); ax[0].legend(); ax[0].set_title("Perdas SSL (ronda 0)")
    for c in ["macro_fg_dice", "dice_fibrin", "dice_granulation", "dice_callus"]:
        ax[1].plot(h.epoch, h[f"val_{c}"], label=c)
    ax[1].axhline(r["val_before"]["macro_fg_dice"], color="k", ls=":", label="macro antes do SSL"); ax[1].legend(fontsize=7); ax[1].set_title("Dice de validação (teacher)")
    for c in CLASS_NAMES:
        ax[2].plot(h.epoch, h[f"kept_{c.lower()}"], label=c)
    ax[2].set_title("Fração de píxeis retidos por classe"); ax[2].legend(fontsize=7)
    for c in CLASS_NAMES:
        ax[3].plot(h.epoch, h[f"share_{c.lower()}"], label=c)
    ax[3].set_title("Proporção prevista por classe nos não anotados\n(deriva = sinal de viés de confirmação)"); ax[3].set_yscale("log"); ax[3].legend(fontsize=7)
    fig.tight_layout(); savefig(fig, "ssl_diagnostics.png"); plt.show()
