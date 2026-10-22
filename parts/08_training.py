# %% [markdown]
# ## 13. Motor de treino e validação
#
# AdamW com **LR diferencial** (encoder ≪ decoder), *warmup* linear + cosseno, **AMP fp16** com `GradScaler`, *gradient clipping*,
# **EMA** dos pesos (o modelo avaliado e guardado é o EMA), *freeze* inicial do encoder pré-treinado (*fine-tuning* progressivo),
# *early stopping* por `macro_fg_dice` de validação, *checkpointing* com **retoma** (`last.pt`) e sondagem automática de OOM
# (reduz o micro-batch e compensa com acumulação, mantendo o batch efetivo). A validação é calculada no **frame de avaliação**
# (tamanho original do conteúdo), com probabilidades revertidas por interpolação bilinear.

# %%
AUTOCAST = lambda: torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP)  # noqa: E731
TTA_SETS = {
    "none": [(0, 0, 1.0)],
    "flip": [(h, v, 1.0) for h in (0, 1) for v in (0, 1)],
    "flip_ms": [(h, v, s) for h in (0, 1) for v in (0, 1) for s in (0.85, 1.0, 1.2)],
}


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.steps = decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))  # warm-up do decay
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)


@torch.no_grad()
def canvas_probs(model: nn.Module, x: torch.Tensor, tta: str) -> torch.Tensor:
    """Média das probabilidades softmax sobre as transformações reversíveis do TTA. x: (B,3,S,S) em [0,1] -> (B,K,S,S)."""
    S, acc, n = x.shape[-1], 0, 0
    for fh, fv, s in TTA_SETS[tta]:
        xi = x
        if fh:
            xi = xi.flip(3)
        if fv:
            xi = xi.flip(2)
        side = max(32, int(round(S * s / 32)) * 32)
        if side != S:
            xi = F.interpolate(xi, size=(side, side), mode="bilinear", align_corners=False)
        with AUTOCAST():
            lg = model(xi)
        p = lg.float().softmax(1)
        if side != S:
            p = F.interpolate(p, size=(S, S), mode="bilinear", align_corners=False)
        if fv:
            p = p.flip(2)
        if fh:
            p = p.flip(3)
        acc, n = acc + p, n + 1
    return acc / n


@torch.no_grad()
def predict_probs(models: list[nn.Module], items: EvalItems, tta: str = "none", weights: list[float] | None = None, bs: int = 8) -> list[np.ndarray]:
    """Probabilidades (K,h,w) float32 por imagem, no frame de avaliação, médias (ponderadas) sobre modelos e TTA."""
    weights = weights or [1.0] * len(models)
    wsum = float(sum(weights))
    out = []
    for m in models:
        m.eval()
    for i in range(0, len(items), bs):
        x = items.canvas[i:i + bs].to(DEVICE, non_blocking=True)
        p = sum(w * canvas_probs(m, x, tta) for m, w in zip(models, weights)) / wsum
        for j in range(p.shape[0]):
            out.append(unletterbox_probs(p[j].cpu(), items.geoms[i + j]).numpy().astype(np.float32))
    return out


def remove_small_components(pred: np.ndarray, min_area_frac: float) -> np.ndarray:
    """Remove componentes conexas de tecido com área < frac da área da imagem; o rótulo de substituição é a moda dos vizinhos (ou fundo)."""
    if min_area_frac <= 0:
        return pred
    out, thr = pred.copy(), min_area_frac * pred.size
    for c in TISSUES:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((out == c).astype(np.uint8), connectivity=8)
        for comp in range(1, n):
            if stats[comp, cv2.CC_STAT_AREA] < thr:
                region = lab == comp
                ring = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & ~region
                vals = out[ring & (out != c)]
                out[region] = np.bincount(vals, minlength=K).argmax() if vals.size else 0
    return out


def probs_to_preds(probs: list[np.ndarray], min_area_frac: float = 0.0) -> list[np.ndarray]:
    return [remove_small_components(p.argmax(0).astype(np.uint8), min_area_frac) for p in probs]


def evaluate_models(models, items: EvalItems, tta: str = "none", weights=None, min_area_frac: float = 0.0):
    probs = predict_probs(models, items, tta, weights)
    return evaluate_preds(items.names, items.gts, probs_to_preds(probs, min_area_frac)), probs


@torch.no_grad()
def canvas_loss(model: nn.Module, items: EvalItems, criterion: nn.Module, bs: int = 8) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(items), bs):
        x, y = items.canvas[i:i + bs].to(DEVICE), items.canvas_gt[i:i + bs].to(DEVICE)
        with AUTOCAST():
            lg = model(x)
        tot += float(criterion(lg.float(), y)) * len(x)
        n += len(x)
    return tot / max(n, 1)


def set_encoder_trainable(model: Net, flag: bool) -> None:
    for p in model.encoder_params():
        p.requires_grad_(flag)


def probe_micro_batch(model: Net, S: int, target: int) -> int:
    """Maior micro-batch (potência de 2 <= target) que executa fwd+bwd sem OOM. Em CPU devolve `target`."""
    if DEVICE.type != "cuda":
        return target
    backup = copy.deepcopy(model.state_dict())
    bs = target
    while bs >= 1:
        try:
            model.train()
            x = torch.rand(bs, 3, S, S, device=DEVICE)
            with AUTOCAST():
                lg = model(x)
            lg.float().mean().backward()
            model.zero_grad(set_to_none=True)
            model.load_state_dict(backup)
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            bs //= 2
    raise RuntimeError("Nem micro-batch 1 cabe na GPU: reduza `size`.")


def lr_factor(step: int, warm: int, total: int, min_ratio: float) -> float:
    if step < warm:
        return (step + 1) / max(warm, 1)
    t = (step - warm) / max(total - warm, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def run_id_of(spec: dict) -> str:
    h = hashlib.md5(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{spec['exp']}__{spec['arch']}__s{spec['seed']}__{h}"


def default_spec(**over) -> dict:
    s = dict(exp="sup", arch=CFG["archs"][0], seed=CFG["seeds"][0], frame=CFG["frame"], size=CFG["size"], loss=CFG["loss"], epochs=CFG["epochs"],
             patience=CFG["patience"], pretrained=True, aug="standard", sampler=True, lr_decoder=CFG["lr_decoder"], lr_encoder=CFG["lr_encoder"],
             freeze_epochs=CFG["freeze_epochs"], batch_size=CFG["batch_size"], keep_ckpt=True)
    s.update(over)
    return s


def _save_half(state: dict, path: Path) -> None:
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}, path)


def _load_half(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}


def _train_run(spec: dict, micro_bs_hint: int | None = None) -> dict:
    rid = run_id_of(spec)
    rdir = DIRS["runs"] / rid
    rdir.mkdir(parents=True, exist_ok=True)
    seed_everything(spec["seed"])
    S = spec["size"]
    recs = get_frame(spec["frame"])
    train_set = SegSet(recs["train"], S, aug=spec["aug"])
    val_items = EvalItems(recs["val"], S)
    loader = make_train_loader(train_set, spec["batch_size"], CFG["fibrin_boost"] if spec["sampler"] else 0.0, spec["seed"])
    model, prov = build_model(spec["arch"], spec["pretrained"])
    model.to(DEVICE)
    micro_bs = micro_bs_hint or probe_micro_batch(model, S, spec["batch_size"])
    criterion = SegLoss(spec["loss"], class_weights_from(recs["train"], CFG["class_weight_cap"]), CFG["label_smoothing"]).to(DEVICE)
    opt = torch.optim.AdamW([{"params": model.encoder_params(), "lr": spec["lr_encoder"]}, {"params": model.other_params(), "lr": spec["lr_decoder"]}],
                            weight_decay=CFG["weight_decay"])
    spe = len(loader)
    total, warm = spec["epochs"] * spe, CFG["warmup_epochs"] * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: lr_factor(st, warm, total, CFG["min_lr_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    ema = EMA(model, CFG["ema_decay"])
    hist, best, best_epoch, bad, start = [], -1.0, -1, 0, 0

    last = rdir / "last.pt"
    if last.exists() and CFG["resume"]:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"]); ema.steps = ck["ema_steps"]
        hist, best, best_epoch, bad, start = ck["hist"], ck["best"], ck["best_epoch"], ck["bad"], ck["epoch"] + 1
        print(f"  [retoma] {rid} a partir da época {start}")

    t0, status = time.time(), "finished"
    for epoch in range(start, spec["epochs"]):
        if not BUDGET.ok(30):
            status = "stopped_by_time_budget"
            break
        freeze = spec["pretrained"] and epoch < spec["freeze_epochs"]
        set_encoder_trainable(model, not freeze)
        model.train()
        run_loss, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            while True:  # repete o batch com micro-batch menor se houver OOM
                try:
                    opt.zero_grad(set_to_none=True)
                    loss_tot = 0.0
                    for xc, yc in zip(xb.split(micro_bs), yb.split(micro_bs)):
                        with AUTOCAST():
                            lg = model(xc)
                        loss = criterion(lg.float(), yc) * (len(xc) / len(xb))
                        scaler.scale(loss).backward()
                        loss_tot += float(loss)
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if micro_bs == 1:
                        raise
                    micro_bs //= 2
                    print(f"  OOM -> micro-batch {micro_bs}")
            if not math.isfinite(loss_tot):
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            run_loss, nb = run_loss + loss_tot, nb + 1
        res, _ = evaluate_models([ema.module], val_items, "none")
        vm = res.metrics()
        score = 0.0 if np.isnan(vm["macro_fg_dice"]) else vm["macro_fg_dice"]
        hist.append(dict(epoch=epoch, train_loss=run_loss / max(nb, 1), val_loss=canvas_loss(ema.module, val_items, criterion),
                         val_macro_fg_dice=vm["macro_fg_dice"], val_micro_fg_dice=vm["micro_fg_dice"], val_macro_fg_iou=vm["macro_fg_iou"],
                         val_paper_dice=vm["paper_dice"], val_dice_fibrin=vm["dice_fibrin"], val_dice_granulation=vm["dice_granulation"],
                         val_dice_callus=vm["dice_callus"], lr_dec=opt.param_groups[1]["lr"]))
        if score > best + 1e-4:
            best, best_epoch, bad = score, epoch, 0
            _save_half(ema.module.state_dict(), rdir / "best.pt")
        else:
            bad += 1
        if (epoch + 1) % 5 == 0:  # ponto de retoma (best.pt é gravado sempre que a validação melhora)
            torch.save(dict(model=model.state_dict(), ema=ema.module.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                            ema_steps=ema.steps, hist=hist, best=best, best_epoch=best_epoch, bad=bad, epoch=epoch), last)
        if epoch % max(1, spec["epochs"] // 10) == 0 or epoch == spec["epochs"] - 1:
            h = hist[-1]
            print(f"  {rid} ep{epoch:03d} loss {h['train_loss']:.3f}/{h['val_loss']:.3f} valDice(macro-fg) {h['val_macro_fg_dice']:.4f} best {best:.4f}@{best_epoch}")
        if bad >= spec["patience"]:
            status = "early_stopped"
            break

    # reavalia o MELHOR checkpoint carregado de disco (fp16->fp32), para que o resultado seja o do artefacto guardado
    final = Net(ARCH_SPECS[spec["arch"]]["ctor"](None)).to(DEVICE)
    final.load_state_dict(_load_half(rdir / "best.pt"))
    res, _ = evaluate_models([final], val_items, "none")
    result = dict(spec=spec, run_id=rid, status=status, best_epoch=best_epoch, epochs_run=len(hist), val=res.metrics(), history=hist, provenance=prov,
                  params_M=round(count_params(model), 2), micro_batch=micro_bs, train_seconds=round(time.time() - t0, 1), ckpt=str(rdir / "best.pt"),
                  device=HW["gpu_name"] or "cpu")
    (rdir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    if last.exists():
        last.unlink()
    if not spec["keep_ckpt"]:
        (rdir / "best.pt").unlink(missing_ok=True)
        result["ckpt"] = None
    del model, ema, opt, final
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def train_run(spec: dict) -> dict:
    """Treina (ou reutiliza, se já existir `result.json`) uma execução. Idempotente: permite reiniciar a sessão do notebook."""
    rid = run_id_of(spec)
    rf = DIRS["runs"] / rid / "result.json"
    if rf.exists() and CFG["resume"]:
        print(f"[reutilizado] {rid}")
        return json.loads(rf.read_text())
    print(f"[treino] {rid}")
    return _train_run(spec)


# %% [markdown]
# ## 14a. Verificação prévia: o modelo consegue memorizar? (sanity check) e *LR range test*
# Antes de gastar horas, confirma-se que o pipeline (dados→modelo→perda→métricas) **consegue sobreajustar 8 imagens sem augmentation**.
# Se não conseguir, há um erro de pipeline, não de hiperparâmetros. O *LR range test* é só consultivo (não altera `CFG`).

# %%
def sanity_overfit(arch: str, n_img: int = 8, steps: int = 80, thr: float = 0.85) -> float:
    seed_everything(0)
    S, recs = CFG["size"], get_frame(CFG["frame"])
    sub = recs["train"][:n_img]
    ds, items = SegSet(sub, S, aug="none"), EvalItems(sub, S)
    loader = torch.utils.data.DataLoader(ds, batch_size=len(sub), shuffle=True, num_workers=0)
    model, _ = build_model(arch, pretrained=True)
    model.to(DEVICE).train()
    crit = SegLoss("ce_dice", class_weights_from(sub, CFG["class_weight_cap"])).to(DEVICE)
    opt = torch.optim.AdamW([{"params": model.encoder_params(), "lr": 3e-4}, {"params": model.other_params(), "lr": 2e-3}], weight_decay=0.0)
    mb = probe_micro_batch(model, S, len(sub))
    it = 0
    while it < steps:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            for xc, yc in zip(xb.split(mb), yb.split(mb)):
                with AUTOCAST():
                    lg = model(xc)
                (crit(lg.float(), yc) * len(xc) / len(xb)).backward()
            opt.step(); it += 1
            if it >= steps:
                break
    res, _ = evaluate_models([model], items, "none")
    d = res.metrics()["macro_fg_dice"]
    verdict = "OK" if d >= thr else ("AVISO (modo smoke: poucos passos)" if MODE == "smoke" else "FALHA")
    print(f"Sanity overfit [{arch}] {n_img} imagens, {steps} passos: macro-fg Dice (treino) = {d:.3f} (limiar {thr}) -> {verdict}")
    del model, opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    if MODE != "smoke" and d < thr:
        raise RuntimeError(f"O modelo {arch} não consegue memorizar {n_img} imagens (Dice {d:.3f}<{thr}). Há um problema de pipeline: não prossiga.")
    return d


def lr_range_test(arch: str, steps: int = 60, lo: float = 1e-6, hi: float = 1e-1) -> float:
    seed_everything(0)
    S, recs = CFG["size"], get_frame(CFG["frame"])
    ds = SegSet(recs["train"], S, aug="standard")
    loader = torch.utils.data.DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True, num_workers=0, drop_last=True)
    model, _ = build_model(arch, pretrained=True)
    model.to(DEVICE).train()
    crit = SegLoss("ce_dice", class_weights_from(recs["train"], CFG["class_weight_cap"])).to(DEVICE)
    opt = torch.optim.AdamW([{"params": model.encoder_params(), "lr": lo * 0.2}, {"params": model.other_params(), "lr": lo}], weight_decay=0.0)
    mult, lrs, losses, sm, it = (hi / lo) ** (1 / steps), [], [], None, 0
    while it < steps:
        for xb, yb in loader:
            lr = lo * mult ** it
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr * 0.2, lr
            opt.zero_grad(set_to_none=True)
            with AUTOCAST():
                lg = model(xb.to(DEVICE))
            loss = crit(lg.float(), yb.to(DEVICE))
            loss.backward(); opt.step()
            sm = float(loss) if sm is None else 0.8 * sm + 0.2 * float(loss)
            lrs.append(lr); losses.append(sm); it += 1
            if it >= steps or not math.isfinite(sm) or sm > 4 * min(losses):
                steps = it
                break
    best_lr = lrs[int(np.argmin(losses))] / 10
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.semilogx(lrs, losses); ax.axvline(best_lr, color="r", ls="--", label=f"mín/10 = {best_lr:.1e}"); ax.legend(); ax.set_xlabel("LR do decoder"); ax.set_ylabel("loss (suavizada)")
    ax.set_title(f"LR range test — {arch}"); fig.tight_layout(); savefig(fig, f"lr_range_{arch}.png"); plt.show()
    print(f"LR sugerido (consultivo) para o decoder: {best_lr:.1e} | configurado: {CFG['lr_decoder']:.1e}")
    del model, opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_lr


if CFG["do_sanity"]:
    SANITY = {a: sanity_overfit(a) for a in CFG["archs"][:1]}
if CFG["do_lr_range"]:
    LR_SUGGESTED = lr_range_test(CFG["archs"][0])
