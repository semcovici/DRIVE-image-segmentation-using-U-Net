import numpy as np
import matplotlib.pyplot as plt



# Função auxiliar para visualização de dados
def visualize(**images):
    """Plotar imagens em uma linha."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.show()
    
# Função auxiliar para visualização de dados    
def denormalize(x):
    """Escala a imagem para o intervalo 0..1 para plotagem correta"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x