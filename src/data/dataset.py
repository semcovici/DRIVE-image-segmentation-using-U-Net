
import cv2
import os
import numpy as np


# Classes para carregamento e pré-processamento de dados
class Dataset:
    """Dataset DRIVE. Lê imagens, aplica augmentations e transformações de pré-processamento.
    
    Args:
        images_dir (str): caminho para a pasta de imagens
        masks_dir (str): caminho para a pasta de máscaras de segmentação
        ids (list): lista de IDs das imagens
        augmentation (albumentations.Compose): pipeline de transformação de dados 
            (por exemplo, flip, scale, etc.)
        preprocessing (albumentations.Compose): pré-processamento de dados 
            (por exemplo, normalização, manipulação de forma, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            ids, 
            dataset_name=None,
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = ids
        self.images_fps = [os.path.join(images_dir, image_id + f'_{dataset_name}.png') for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id + '_manual1.png') for image_id in self.ids]
        
        # Verificar se os arquivos existem
        for img_fp, mask_fp in zip(self.images_fps, self.masks_fps):
            if not os.path.exists(img_fp):
                print(f"Imagem não encontrada: {img_fp}")
            if not os.path.exists(mask_fp):
                print(f"Máscara não encontrada: {mask_fp}")
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Ler dados
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        
        # Normalizar máscara para [0,1]
        mask = mask / 255.0
        mask = mask.astype('float32')
        mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)
        
        # Aplicar augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Aplicar pré-processamento
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
    
    def __len__(self):
        return len(self.ids)
