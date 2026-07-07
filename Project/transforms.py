# transforms.py
# Transformacoes de imagem para modelos pre-treinados no ImageNet

import torch
from torchvision import transforms


class ImageTransforms:
    """
    Transformações, normalização e filtros de imagem para modelos pré-treinados.

    Todos os backbones (VGG, ResNet, ConvNeXt) foram treinados com imagens
    normalizadas com a média e desvio padrão do ImageNet. É obrigatório usar
    os mesmos valores:

        média  = [0.485, 0.456, 0.406]
        desvio = [0.229, 0.224, 0.225]

    Além disso, os modelos esperam entradas de 224x224 pixels.
    """

    # parâmetros de normalização do ImageNet
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    @staticmethod
    def get_train_transform():
        """
        Transformações com augmentação para o conjunto de treino.
        A augmentação aumenta a diversidade sem coletar novos dados.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(ImageTransforms.IMAGENET_MEAN, ImageTransforms.IMAGENET_STD),
        ])

    @staticmethod
    def get_val_transform():
        # Transformações padrão para validação e teste (sem augmentação).
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(ImageTransforms.IMAGENET_MEAN, ImageTransforms.IMAGENET_STD),
        ])

    @staticmethod
    def denormalize(tensor):
        """
        Inverte a normalização do ImageNet para visualização de imagens.
        Necessário para exibir imagens sem o artefato da normalização.
        Uso: img = ImageTransforms.denormalize(img_tensor)
        """
        mean_t = torch.tensor(ImageTransforms.IMAGENET_MEAN).view(3, 1, 1)
        std_t  = torch.tensor(ImageTransforms.IMAGENET_STD).view(3, 1, 1)
        return (tensor * std_t + mean_t).clamp(0, 1)
