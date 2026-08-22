import random
import matplotlib.pyplot as plt


class IJDEVLibs:
    """
    Biblioteca utilitária criada por Idelfrides para organização e reutilização de recursos.
    """
    
    default_cmaps = [] 
    
    def __init__(self):
        IJDEVLibs.default_cmaps = ['Paired', 'grey', 'Reds']
        
     
    @classmethod
    def get_random_cmap(cls) -> str:
        """
        Retorna um colormap aleatório entre 'Paired', 'grey' e 'Reds'.
        """
        return random.choice(cls.default_cmaps)
    
    
    @staticmethod
    def get_random_choice(options: list) -> str:
        """
        Retorna um elemento aleatório da lista fornecida.
        
        Parâmetros:
        - options: lista de strings ou valores
        
        Exemplo:
        >>> IJDEVLibs.get_random_choice(['Paired', 'grey', 'Reds'])
        'grey'
        """
        return random.choice(options)


    @staticmethod
    def smart_make_plot(x_data, y_data, ncols: int = 5, nrows: int = 1,
                     sharex: bool = False, sharey: bool = True,
                     figsize: tuple = (10, 4), cmap: str = 'gray') -> None:
        """
        Exibe imagens lado a lado com títulos.
        
        Parâmetros:
        - x_data: array ou lista com imagens
        - y_data: array ou lista com rótulos/títulos
        - ncols: número de colunas
        - nrows: número de linhas
        - sharex/sharey: controle de eixos
        - figsize: tamanho da figura
        - cmap: colormap para imshow
        """
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 sharex=sharex, sharey=sharey,
                                 figsize=figsize)

        axes = axes.flatten()  # garante array 1D

        for i in range(min(len(x_data), ncols * nrows)):
            axes[i].imshow(x_data[i], cmap=cmap)
            axes[i].set_title(str(y_data[i]))
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


    @staticmethod
    def get_deep_learning_best_acc(model_history: dict) -> tuple:
        """
        Recupera o maior valor de acurácia de validação (val_acc) 
        e a época correspondente a partir do histórico de treinamento.

        Parâmetros:
        - model_history (dict): dicionário retornado por history.history do Keras

        Retorno:
        - Tuple (best_epoch, max_val_acc)
          best_epoch (int): número da época onde ocorreu o melhor val_acc
          max_val_acc (float): maior acurácia de validação registrada
        """
        val_acc_list = model_history['val_acc']
        max_val_acc = max(val_acc_list)
        best_epoch = val_acc_list.index(max_val_acc) + 1  # +1 porque épocas começam em 1
        return best_epoch, max_val_acc