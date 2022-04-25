import numpy as np


class Perceptron(object):

    """Clasificador perceptron

    Parametros
    ----------

    eta: float
        Ratio de aprendizaje (0.0 a 1.0)

    n_iter: int
        Numero de iteraciones sobre el dataset

    random_state: int
        Semilla generadora de numeros aleatorios 
        para inicializar los pesos.


    Atributos
    ---------

    w_: 1d-array
        Pesos despues de entrenar
    
    errors_: list
        Numeros de clasificaciones incorrectas 
        en cada iteracion

    """

    def __init__(self, eta= 0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Entrenar data
        
        Parametros
        ---------

        X: [array], shape =[n_muestras, n_features]
            Vector de entrenamiento
        
        Return: object

        """

        #Generador de numeros aleatorios con una distribucion normal
        # desviacion estandar 0.01

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= 1 + X.shape[1])

        self.errors_ = []

        #Iterar sobre el dataset
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        
        return self

    def net_input(self, X):
        """Calcular el net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Retorna label de clase"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)