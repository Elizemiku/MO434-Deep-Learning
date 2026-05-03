# kd_utils.py
# Ponto de entrada unico do projeto KD - MO434
#
# Fachada que re-exporta tudo dos modulos especializados.
# O notebook importa apenas deste arquivo; nada precisa mudar la.
#
# Estrutura de modulos:
#   kd_helpers.py      -> set_seed, Timer
#   transforms.py      -> ImageTransforms
#   datasets.py        -> DatasetManager
#   models/
#     blocks.py        -> conv_block, ResBlock
#     encoders.py      -> PlainCNNEncoder, DepthwiseCNNEncoder, MiniResNetEncoder
#     teacher.py       -> TeacherWrapper
#     predictors.py    -> PreditorPostGAP, PreditorPreGAP, StudentModel
#   losses.py          -> PerdaKD, PerdaRKD
#   trainer.py         -> Trainer
#   evaluator.py       -> Evaluator
#   visualization.py   -> plotar_curvas, plotar_comparacao_mse_rkd, plotar_graficos_analise

from kd_helpers import set_seed, Timer

from transforms import ImageTransforms
from datasets import DatasetManager

from models import (
    conv_block, ResBlock,
    PlainCNNEncoder, DepthwiseCNNEncoder, MiniResNetEncoder,
    TeacherWrapper,
    PreditorPostGAP, PreditorPreGAP, StudentModel,
)

from losses import PerdaKD, PerdaRKD
from trainer import Trainer
from evaluator import Evaluator
from visualization import plotar_curvas, plotar_comparacao_mse_rkd, plotar_graficos_analise

__all__ = [
    'set_seed', 'Timer',
    'ImageTransforms', 'DatasetManager',
    'conv_block', 'ResBlock',
    'PlainCNNEncoder', 'DepthwiseCNNEncoder', 'MiniResNetEncoder',
    'TeacherWrapper',
    'PreditorPostGAP', 'PreditorPreGAP', 'StudentModel',
    'PerdaKD', 'PerdaRKD',
    'Trainer', 'Evaluator',
    'plotar_curvas', 'plotar_comparacao_mse_rkd', 'plotar_graficos_analise',
]
