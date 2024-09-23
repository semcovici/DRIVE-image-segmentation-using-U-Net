import numpy as np
import keras


class Dataloder(keras.utils.Sequence):
    """Carrega dados do dataset e forma batches
    
    Args:
        dataset: instância da classe Dataset para carregamento e pré-processamento de imagens.
        batch_size: Número inteiro de imagens no batch.
        shuffle: Booleano, se `True` embaralha os índices das imagens a cada época.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        # Coletar dados do batch
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # Transpor lista de listas
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denota o número de batches por época"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Função callback para embaralhar os índices a cada época"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   
