# Chama a estrutura de modulos:
#   kd_utils.py        -> set_seed, Timer
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

from Project.kd_utils import set_seed, Timer

from transforms import ImageTransforms
from load_datasets import DatasetManager

from Project.models import (
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
