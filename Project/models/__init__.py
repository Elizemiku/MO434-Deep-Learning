# models/__init__.py
# Re-exporta todos os componentes do pacote models para importacao direta.
#
# Uso:
#   from models import TeacherWrapper, PlainCNNEncoder, StudentModel, ...

from models.blocks import conv_block, ResBlock
from models.encoders import PlainCNNEncoder, DepthwiseCNNEncoder, MiniResNetEncoder
from models.teacher import TeacherWrapper
from models.predictors import PreditorPostGAP, PreditorPreGAP, StudentModel

__all__ = [
    'conv_block', 'ResBlock',
    'PlainCNNEncoder', 'DepthwiseCNNEncoder', 'MiniResNetEncoder',
    'TeacherWrapper',
    'PreditorPostGAP', 'PreditorPreGAP', 'StudentModel',
]
