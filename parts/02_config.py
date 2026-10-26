# %% [markdown]
# ## 5. Configuração global
#
# Escolha **`MODE`**:
# * `smoke` — diagnóstico de pipeline (minutos, mesmo em CPU; modelo pequeno com pesos aleatórios; **os números não têm valor científico**);
# * `quick` — primeira corrida real em GPU (2 arquiteturas, 1 semente, sem sondas nem ablação);
# * `full` — experiência completa: **validação cruzada de 5 folds** nas 94 imagens de desenvolvimento com 3 arquiteturas, sondas de *frame*/resolução (256–512) e de perda,
#   SSL em 2 rondas sobre todos os folds das 2 melhores arquiteturas (avaliado sem seleção de época), ablação, ajuste de pós-processamento no OOF e *bootstrap* de 2000.
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
    probe_epochs=40, ablation_epochs=40, probe_frames=[("padded", 256), ("roi", 256), ("roi", 384), ("roi", 512)],
    # validação cruzada nas 94 imagens de desenvolvimento (treino+val oficiais; o teste nunca entra)
    do_cv=False, cv_folds=5, cv_seed=0,
    cv_epochs={"unetpp_effb4": 130, "upernet_convnext_s": 70, "segformer_mitb3": 70, "upernet_convnext_b": 70, "segformer_mitb5": 70, "smoke_fpn_r18": 2},  # épocas fixas (EMA final, sem seleção) -> OOF sem viés
    probe_losses=["ce_dice", "focal_dice", "focal_tversky", "ce_dice_lovasz"],
    # SSL
    do_ssl=True, ssl_rounds=2, ssl_epochs=40, ssl_steps_per_epoch=20, ssl_lambda_u=1.0, ssl_keep_frac=0.7,
    ssl_tau_min=0.75, ssl_tau_max=0.97, ssl_dup_hamming=8,
    ssl_select="best",      # 'best': melhor época na validação (com rollback) | 'last': pesos finais, sem seleção (avaliação não enviesada)
    ssl_archs_top=1,        # nº de arquiteturas (melhores no OOF/val) a que se aplica SSL; todos os seus modelos/folds são usados
    ssl_ema=True,           # SSL EMA online (professor = EMA do próprio aluno): ~19 min por modelo em T4; o `ssle` dá o mesmo ou mais e custa menos
    ssl_ens=False,          # SSL com professor ENSEMBLE limpo por fold (pseudo-labels offline do ensemble de todas as arquiteturas do mesmo fold, que não viram as imagens retidas)
    ssl_ens_tta="flip",
    ssl_ens_gens=1,         # gerações: g=2 usa como professor o ensemble {supervisionados + alunos da geração 1} do mesmo fold; o aluno parte sempre do supervisionado
    # inferência / estatística
    tta_candidates=["none", "flip", "flip_ms"], pp_min_area_fracs=[0.0, 0.0005, 0.001, 0.002, 0.005],
    pp_fibrin_fracs=[0.0, 0.001, 0.002, 0.005, 0.01], pp_scale_grid=[0.6, 0.8, 1.0, 1.25, 1.6], tune_class_scale=True, pp_gain_margin=0.002, pp_fibrin_conf=[0.0, 0.5, 0.6, 0.7, 0.8], deterministic=True,
    bootstrap_b=2000, paired_bootstrap_b=1000, time_budget_hours=9.0, resume=True, seed_base=42,
)
_MODES = {
    "smoke": dict(archs=["smoke_fpn_r18"], seeds=[0], epochs=3, patience=3, warmup_epochs=1, freeze_epochs=1, batch_size=8,
                  ssl_rounds=1, ssl_epochs=2, ssl_steps_per_epoch=3, bootstrap_b=100, paired_bootstrap_b=50, do_lr_range=True, num_workers=0,
                  do_cv=True, cv_folds=2, ssl_select="last", pp_scale_grid=[0.8, 1.0, 1.25], ssl_ens=True, ssl_ens_gens=2,
                  do_probes=True, probe_epochs=1, probe_frames=[("padded", 256), ("roi", 256)],
                  probe_losses=["ce_dice", "focal_tversky"], do_ablation=True, ablation_epochs=1, time_budget_hours=2.0),
    "quick": dict(archs=["unetpp_effb4", "segformer_mitb3"], seeds=[42], epochs=60, patience=20, ssl_rounds=1, ssl_epochs=30,
                  bootstrap_b=1000, time_budget_hours=4.0),
    # `full`: arquiteturas por ordem de prioridade (se o tempo acabar, as primeiras ficam com todos os folds). A EfficientNet-B4 foi retirada (pior por ~4 pp, IC robusto, na corrida anterior);
    # frame/resolução fixados em roi/384 e sondas desligadas (a corrida anterior deu empate entre perdas e 384>512). Ligue-as com DFU_CFG_OVERRIDE se quiser re-verificar.
    # Corrida anterior (OOF, 94 imagens): ConvNeXt-B e MiT-B5 não melhoraram (B5 pior, IC robusto) -> retirados; SSL EMA custou 3,1 h e o `ssle` igualou-o ou superou-o -> desligado;
    # o Dice do SSL estabiliza por volta da época 10 -> 20 épocas bastam. 2.ª corrida: a geração 2 do professor ensemble não acrescentou nada (Δ≈0, indistinguível) e custou 175 min ->
    # volta a 1 geração, e o tempo vai para uma 2.ª semente (mais membros no ensemble; o OOF faz a média dos membros que não viram cada imagem).
    "full": dict(archs=["upernet_convnext_s", "segformer_mitb3"], seeds=[42, 43], size=384, do_probes=False, do_ablation=False, do_cv=True,
                 ssl_select="last", ssl_archs_top=2, ssl_ema=False, ssl_ens=True, ssl_ens_gens=1, ssl_epochs=20, time_budget_hours=10.5),
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
    if CFG.get("deterministic", True):
        torch.use_deterministic_algorithms(True, warn_only=True)   # operações sem versão determinística só avisam (avisos silenciados)


seed_everything(CFG["seed_base"])
print(f"MODE={MODE} | saídas em {OUT}")
print(json.dumps({k: v for k, v in CFG.items() if k in ["archs", "seeds", "frame", "size", "epochs", "batch_size", "loss", "do_probes", "do_ssl"]}))
