"""
Module Doc Strings
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from numpy.random import multinomial
from scipy.stats import invgamma, norm, multivariate_normal

class FiniteGMM:
    """
    Creates a FiniteGMM class for fitting Univariate and Multivariate Gaussian
    Mixture Models.

    Args:
        k (int): The number of Gaussians in the mixture
        mu (np.array): The mean of the Gaussians
        sigma (np.array): The covariance of the Gaussians
        phi (np.array): The mixing coefficients
    Returns:
        FiniteGMM: A FiniteGMM object
    """
    def __init__(self, k = None, mu = None, sigma = None, phi = None):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.multivariate = np.array(self.sigma).ndim >= 2
        self.data = np.NaN

    def data_gen(self, n):
        """
        Generates samples from Mixture of K Gaussian Distributions. Method will generate multivariate
        or univariate data depending on the shape of the sigma parameter. Data is assigned to class attribute "data"

        Args:
            N (int): The number of samples you want to generate
        Returns:
            None, but assigns data to class attribute "data"
        """
        try:
            y = np.empty((n, np.array(self.mu).shape[1]))
        except IndexError:
            y = np.empty(n)
        z = []
        for i in range(n):
            ind = multinomial(1, self.phi)
            for j, val in enumerate(ind):
                if val == 1:
                    if self.multivariate:
                        z.append(j)
                        y[i,:] = np.random.multivariate_normal(self.mu[j,:], self.sigma[j,:,:])
                    else:
                        y[i] = norm(self.mu[j], self.sigma[j]).rvs()
        self.data = np.array(y)
        return

    def plot_data(self, **kwargs):
        """
        Plots the original data as histogram for univariate and scatter plot for multivariate data
        
        Args:
            kwargs: Keyword arguments to be passed to matplotlib.pyplot.hist() for univariate data
            or matplotlib.pyplot.scatter() for multivariate data
        """
        if type(self.data) != np.ndarray:
            raise ValueError("You must generate data first!")
        
        if self.multivariate:
            x, y = np.mgrid[min(self.data[:,0])-1:max(self.data[:,0])+1:.1, min(self.data[:,1])-1:max(self.data[:,1])+1:.1]
            pos = np.dstack((x,y))
            fig, ax = plt.subplots()
            ax.scatter(self.data[:,0], self.data[:,1], **kwargs)
            if self.mu is not None and self.sigma is not None and self.k is not None:
                for i in range(len(self.mu)):
                    ax.contour(x,y, multivariate_normal(self.mu[i,:], self.sigma[i,:,:]).pdf(pos), extend='both')
                    fig.suptitle(f"K={self.k} Bivariate Gaussian Distributions Data")
            else:
                warnings.warn("No True Parameters Given... Just plotting the data")
                plt.subtitle("Finite GMM Data")
            ax.grid()
            fig.show()

        else:
            x_data = np.linspace(min(self.data), max(self.data))
            plt.hist(self.data, density=True, **kwargs)
            if self.mu is not None and self.sigma is not None:
                for i in range(len(self.mu)):
                    print(list(mcolors.TABLEAU_COLORS.values())[i])
                    plt.plot(x_data, norm(self.mu[i], self.sigma[i]).pdf(x_data), color=list(mcolors.TABLEAU_COLORS.values())[i])
                    plt.title(f"Mixture of {len(self.mu)} Gaussians Data")
            else:
                warnings.warn("No True Parameters Given... Just plotting the data")
                plt.title("Finite GMM Data")
            plt.grid()
            plt.show()

if __name__ == '__main__':
    pass
