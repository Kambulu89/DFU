# %% [markdown]
# ## 5. Configuração global
#
# Escolha **`MODE`**:
# * `smoke` — diagnóstico de pipeline (minutos, mesmo em CPU; modelo pequeno com pesos aleatórios; **os números não têm valor científico**);
# * `quick` — primeira corrida real em GPU (2 arquiteturas, 1 semente, sem sondas nem ablação);
# * `full` — experiência completa (3 arquiteturas × 3 sementes, sondas de *frame*/resolução e de perda, ablação, SSL em 2 rondas, *bootstrap* de 2000).
#
# Todos os hiperparâmetros vivem em `CFG`; o dicionário completo, as sementes e as versões são gravados em `config.json`.

# %%
MODE = os.environ.get("DFU_MODE", "quick")  # 'smoke' | 'quick' | 'full'

_BASE = dict(
    # dados
    frame="roi",            # 'roi' (recorta + letterbox do conteúdo) | 'padded' (tela 256x256 do artigo)
    size=256,               # lado da tela de entrada do modelo
    num_workers=min(4, os.cpu_count() or 1),
    # treino
    batch_size=16, epochs=150, patience=40, warmup_epochs=5, freeze_epochs=3,
    lr_decoder=1e-3, lr_encoder=2e-4, weight_decay=1e-2, ema_decay=0.98, grad_clip=1.0, min_lr_ratio=0.01,
    loss="ce_dice", label_smoothing=0.0, fibrin_boost=0.5, class_weight_cap=5.0,
    # fases opcionais
    do_sanity=True, do_lr_range=True, do_probes=False, do_ablation=False,
    probe_epochs=40, ablation_epochs=40, probe_frames=[("padded", 256), ("roi", 256), ("roi", 384)],
    probe_losses=["ce_dice", "focal_dice", "focal_tversky", "ce_dice_lovasz"],
    # SSL
    do_ssl=True, ssl_rounds=2, ssl_epochs=40, ssl_steps_per_epoch=20, ssl_lambda_u=1.0, ssl_keep_frac=0.7,
    ssl_tau_min=0.75, ssl_tau_max=0.97, ssl_all_seeds=False, ssl_dup_hamming=8,
    # inferência / estatística
    tta_candidates=["none", "flip", "flip_ms"], pp_min_area_fracs=[0.0, 0.0005, 0.001, 0.002, 0.005],
    bootstrap_b=2000, time_budget_hours=9.0, resume=True, seed_base=42,
)
_MODES = {
    "smoke": dict(archs=["smoke_fpn_r18"], seeds=[0], epochs=3, patience=3, warmup_epochs=1, freeze_epochs=1, batch_size=8,
                  ssl_rounds=1, ssl_epochs=2, ssl_steps_per_epoch=3, bootstrap_b=100, do_lr_range=True, num_workers=0,
                  do_probes=True, probe_epochs=1, probe_frames=[("padded", 256), ("roi", 256)],
                  probe_losses=["ce_dice", "focal_tversky"], do_ablation=True, ablation_epochs=1, time_budget_hours=2.0),
    "quick": dict(archs=["unetpp_effb4", "segformer_mitb3"], seeds=[42], epochs=60, patience=20, ssl_rounds=1, ssl_epochs=30,
                  bootstrap_b=1000, time_budget_hours=4.0),
    "full": dict(archs=["unetpp_effb4", "upernet_convnext_s", "segformer_mitb3"], seeds=[42, 43, 44], do_probes=True,
                 do_ablation=True, ssl_all_seeds=False),
}
assert MODE in _MODES, f"MODE inválido: {MODE}"
CFG = {**_BASE, **_MODES[MODE], "mode": MODE}
CFG.update(json.loads(os.environ.get("DFU_CFG_OVERRIDE", "{}")))  # sobreposição opcional por variável de ambiente (JSON)

OUT = WORK / f"dfu_run_{MODE}"
DIRS = {k: OUT / k for k in ["runs", "figures", "tables", "models", "cache", "predictions"]}
for _d in DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Background", "Fibrin", "Granulation", "Callus"]
K = 4
TISSUES = [1, 2, 3]
PALETTE = np.array([[0, 0, 0], [230, 30, 30], [40, 200, 60], [40, 90, 240]], dtype=np.uint8)  # bg, fibrina, granulação, calo

# Resultados reportados no artigo (Tabelas 3, 4 e 5) — constantes copiadas, NÃO resultados desta sessão.
PAPER = {
    "supervised": {"IoU": 73.75, "Precision": 87.69, "Recall": 82.27, "Dice": 84.89},
    "semi_supervised": {"IoU": 77.99, "Precision": 88.14, "Recall": 87.14, "Dice": 87.64},
    "per_class_ssl": {
        "Fibrin": {"IoU": 52.68, "Precision": 71.48, "Recall": 66.70, "Dice": 69.01},
        "Granulation": {"IoU": 88.87, "Precision": 93.12, "Recall": 95.11, "Dice": 94.11},
        "Callus": {"IoU": 64.29, "Precision": 80.65, "Recall": 76.02, "Dice": 78.27},
    },
}


class _Budget:
    """Orçamento de tempo de parede (sessões Kaggle têm limite de duração)."""

    def __init__(self, hours: float):
        self.t0, self.limit = time.time(), hours * 3600

    def left(self) -> float:
        return self.limit - (time.time() - self.t0)

    def ok(self, need_s: float = 0.0) -> bool:
        return self.left() > need_s


BUDGET = _Budget(CFG["time_budget_hours"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(CFG["seed_base"])
print(f"MODE={MODE} | saídas em {OUT}")
print(json.dumps({k: v for k, v in CFG.items() if k in ["archs", "seeds", "frame", "size", "epochs", "batch_size", "loss", "do_probes", "do_ssl"]}))
