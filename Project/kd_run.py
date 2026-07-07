# Chama as estruturas de classes pra rodar as funções necessárias para as análises.

from kd_utils import set_seed, Timer

from transforms import ImageTransforms
from load_datasets import DatasetManager

from models import (
    conv_block, ResBlock, SEBlock,
    PlainCNNEncoder, DepthwiseCNNEncoder, MiniResNetEncoder,
    TeacherWrapper,
    PreditorPostGAP, PreditorPreGAP, StudentModel,
)

from losses import PerdaKD, PerdaRKD, PerdaKDCosine, PerdaKDCKA
from trainer import Trainer
from evaluator import Evaluator
from visualization import plotar_curvas, plotar_comparacao_mse_rkd, plotar_graficos_analise

__all__ = [
    'set_seed', 'Timer',
    'ImageTransforms', 'DatasetManager',
    'conv_block', 'ResBlock', 'SEBlock',
    'PlainCNNEncoder', 'DepthwiseCNNEncoder', 'MiniResNetEncoder',
    'TeacherWrapper',
    'PreditorPostGAP', 'PreditorPreGAP', 'StudentModel',
    'PerdaKD', 'PerdaRKD', 'PerdaKDCosine', 'PerdaKDCKA',
    'Trainer', 'Evaluator',
    'plotar_curvas', 'plotar_comparacao_mse_rkd', 'plotar_graficos_analise',
]
