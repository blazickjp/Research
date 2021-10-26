"""
Module Doc Strings
"""

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from numpy.random import multinomial
from scipy.stats import invgamma, norm, dirichlet

class FiniteGMM:
    """
    Creates a FiniteGMM class
    """
    def __init__(self, k, mu, sigma, phi):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.data = np.NaN
    
    def dataGen(self, N):
        """
        Generates samples from Mixture of 2 Gaussian Distributions. Data is assigned to class attribute "data"
        Args:
            N (int): The number of samples you want to generate
        """
        y = []
        for _ in range(N):
            ind = multinomial(1, self.phi)
            for j, val in enumerate(ind):
                if val == 1:
                    y.append(norm(self.mu[j], self.sigma[j]).rvs())
        self.data = np.array(y)
        return

    def plot_data(self, **kwargs):
        """
        Plots the original data as histogram
        """
        x_data = np.linspace(min(self.data), max(self.data))
        plt.hist(self.data, density=True, **kwargs)
        for i in range(len(self.mu)):
            print(list(mcolors.TABLEAU_COLORS.values())[i])
            plt.plot(x_data, norm(self.mu[i], self.sigma[i]).pdf(x_data), color=list(mcolors.TABLEAU_COLORS.values())[i])
        plt.title(f"Mixture of {len(self.mu)} Gaussians Data")
        plt.grid()
        plt.show()



if __name__ == '__main__':
    pass
