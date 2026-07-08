# datasets.py
# Faz o carregamento e gerencia os datasets do projeto

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from transforms import ImageTransforms


class DatasetManager:
    """
    Datasets:
    - flowers102: 102 classes de flores, ~8.000 imagens com resoluções variadas
      splits oficiais: train (1020), val (1020), test (6149)
      ideal para Q1: fine-grained, ConvNeXt tende a se sair melhor

    - pets: Oxford-IIIT-Pet, 37 raças de cães e gatos, ~7.000 imagens
      testa a transferência na classificação de granularidade fina com menos dados
    """

    def __init__(self, data_root='./data', batch_size=32, num_workers=2):
        self.data_root   = data_root
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.datasets    = {}

        # transformações reutilizadas por todos os datasets
        self.transform_train = ImageTransforms.get_train_transform()
        self.transform_val   = ImageTransforms.get_val_transform()

    def load_flowers102(self):
        """
        Carrega Flowers-102 com splits oficiais
        Ideal para Q1: fine-grained, a diferença entre teachers fica mais evidente.
        """
        train_raw = datasets.Flowers102(
            root=self.data_root, split='train', download=True,
            transform=self.transform_train)
        val_raw = datasets.Flowers102(
            root=self.data_root, split='val', download=True,
            transform=self.transform_val)
        test_raw = datasets.Flowers102(
            root=self.data_root, split='test', download=True,
            transform=self.transform_val)

        self.datasets['flowers102'] = {
            'n_classes': 102,
            'train': DataLoader(train_raw, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True),
            'val':   DataLoader(val_raw,   batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True),
            'test':  DataLoader(test_raw,  batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True),
        }
        print(f"flowers-102 -> treino: {len(train_raw):,} | val: {len(val_raw):,} | teste: {len(test_raw):,}")
        return self.datasets['flowers102']

    def load_oxford_pets(self):
        """
        Carrega Oxford-IIIT-Pet com 37 classes.
        Divisão: 80% treino, 20% validação do conjunto trainval.
        Confirma que os resultados generalizam para um domínio diferente do Flowers-102.
        """
        trainval = datasets.OxfordIIITPet(
            root=self.data_root, split='trainval', download=True,
            transform=self.transform_train)
        test_raw = datasets.OxfordIIITPet(
            root=self.data_root, split='test', download=True,
            transform=self.transform_val)

        # divisão do conjunto trainval
        n_train = int(0.8 * len(trainval))
        n_val   = len(trainval) - n_train
        tr, vl  = random_split(trainval, [n_train, n_val])

        self.datasets['pets'] = {
            'n_classes': 37,
            'train': DataLoader(tr,       batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True),
            'val':   DataLoader(vl,       batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True),
            'test':  DataLoader(test_raw, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True),
        }
        print(f"oxford-pets -> treino: {len(tr):,} | val: {len(vl):,} | teste: {len(test_raw):,}")
        return self.datasets['pets']

    def load_all(self):
        # carrega todos os datasets e retorna o mapa de configurações.
        self.load_flowers102()
        self.load_oxford_pets()
        return self.datasets

    def get(self, name):
        # retorna configuração de um dataset pelo nome ('flowers102' ou 'pets').
        if name not in self.datasets:
            raise KeyError(f"dataset '{name}' não carregado. chame load_{name}() primeiro.")
        return self.datasets[name]

    def visualizar_amostras(self, n_imgs=8):
        """
        Visualização: Amostras de imagens dos datasets carregados.
        Inverte a normalização do ImageNet para que as imagens sejam exibidas corretamente.
        """
        nomes = list(self.datasets.keys())
        fig, axes = plt.subplots(len(nomes), n_imgs, figsize=(n_imgs * 2, len(nomes) * 2.5))
        if len(nomes) == 1:
            axes = [axes]

        for row, nome in enumerate(nomes):
            loader = self.datasets[nome]['val']
            imgs, labels = next(iter(loader))
            for i in range(min(n_imgs, len(imgs))):
                img = ImageTransforms.denormalize(imgs[i]).permute(1, 2, 0).numpy()
                axes[row][i].imshow(img)
                axes[row][i].set_title(f"cls {labels[i].item()}", fontsize=8)
                axes[row][i].axis('off')
            axes[row][0].set_ylabel(nome, fontsize=10, rotation=90)

        plt.suptitle("Amostras dos datasets (normalização invertida para visualização)",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
