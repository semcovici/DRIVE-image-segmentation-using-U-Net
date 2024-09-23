import albumentations as A

# Definir augmentations
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.GaussNoise(p=0.2),
        A.Resize(256, 256),
        A.Lambda(name='round_clip_0_1',mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_validation_augmentation():
    test_transform = [
        A.Resize(256, 256),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Constrói a transformação de pré-processamento
    
    Args:
        preprocessing_fn (callable): função de normalização de dados 
            (pode ser específica para cada rede neural pré-treinada)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        A.Lambda(name='pre_processing_fn',image=preprocessing_fn),
    ]
    return A.Compose(_transform)