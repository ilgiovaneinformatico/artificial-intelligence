import numpy as np  # possibile errore sul modulo se non scaricato
import pandas as pd


class Perceptron(object):
    """Perceptron classifier.

    Parametri 
    ------------
    eta : float
        Tasso di apprentimento - 0.0 / 1.0
    n_iter : int
        Epoch - passi all'interno del dataset di apprendimento

    Attributi
    -----------
    w_ : 1d-array
        Pesi dopo inizializzazione
    errors_ : list
        Numero di classificazioni errate (aggiornamenti) in ogni epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Inizializzazione dati apprendimento.

        Parametri
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Vettori di addestramento, dove n_samples è il numero di campioni e
            n_features è il numero di funzionalità.
        y : array-like, shape = [n_samples]
            Valori target. 

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1]) #inizializzazione pesi
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
