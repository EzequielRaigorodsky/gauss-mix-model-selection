#!/home/pi/datascie/bin/python3
# -*- coding: utf-8 -*-
# Ezequiel Raigorodsky ezequielraigorodsky@gmail.com

# %%
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# np.random.seed(42)


class MixModSelection():
    """This class goes over a range of clusters and of covariance types
    to find which one is the best using the BIC. Note that the fit will be
    made on a cross-join of clusters and covaraince types.

    Base estimator is sklearn.mixer.GaussianMixer(), and this class is built
    on top of that one. However, note that this one does NOT inherit methods.
    This was built as a simple workaround to the lack of a class for this on
    sklearn and for simple use. If you need or want to pass specific
    parameters to the estimator, instance it before and pass it as a arg
    when declaring the instance of this class.
    """
    def __init__(
        self,
        estim=GaussianMixture(),
        n_clust=np.arange(2, 10),
        cov_type=['spherical', 'tied', 'diag', 'full'],
        num_init=3,
        rand_state=42
    ):
        self.estimator = estim

        if isinstance(n_clust, int):
            self.n_cluster = np.arange(1, n_clust+1)
        elif isinstance(n_clust, list):
            self.n_cluster = np.array(n_clust)
        elif isinstance(n_clust, np.ndarray):
            if n_clust[0] == 0:
                n_clust = n_clust[1:]
            self.n_cluster = n_clust
        else:
            raise TypeError('n_clust should be either an integer, a list or '
                            'a numpy 1-d array')

        if isinstance(cov_type, str):
            self.cov_type = np.array([cov_type])
        elif isinstance(cov_type, list):
            self.cov_type = np.array(cov_type)
        elif isinstance(cov_type, np.ndarray):
            self.cov_type = cov_type
        else:
            raise TypeError('n_clust should be either an integer, a list or '
                            'a numpy 1-d array')

        if isinstance(num_init, int):
            self.num_init = num_init
        else:
            raise TypeError('num_init has to be an integer')

        if isinstance(rand_state, int):
            self.rand_state = rand_state
        else:
            raise TypeError('rand_state has to be an integer')

        self.is_fit = False
        self.best_params_ = None
        self.best_estimator_ = None
        self.best_bic_ = None
        self.all_means_ = []
        self.all_covariances_ = []
        self.best_stats_ = {}

        self._colors = [
            'xkcd:red', 'xkcd:blue', 'xkcd:green',
            'xkcd:orange', 'xkcd:pink', 'xkcd:purple',
            'xkcd:teal', 'xkcd:lavender', 'xkcd:cyan'
        ]
        self._markers = ['o', '+', '*', 'x', 's', '<', '>', '^', 'v']

    def fit(self, X):
        """Fit the data passed with the estimator passed.
        The fit is applied on the range of clusters and the covariance types.

        Args:
            X (pd.DataFrame or np.ndarray): Data to cluster
        """
        # fit for all n_clust and and cov_type
        self.iter_over = [[n, c] for n in self.n_cluster
                          for c in self.cov_type]
        estimator = self.estimator
        for i, (num, cov) in enumerate(self.iter_over):
            estimator.set_params(
                n_components=num,
                covariance_type=cov,
                n_init=self.num_init,
                random_state=self.rand_state
            )
            estimator.fit(X)
            self.iter_over[i].append(estimator.bic(X))
            self.all_means_.append(
                [num, cov, estimator.means_]
            )
            self.all_covariances_.append(
                [num, cov, estimator.covariances_]
            )

        self.params_score_ = np.array(self.iter_over)
        idx_best = np.argmin(self.params_score_[:, 2])
        self.best_params_ = {
            'n_components': int(self.params_score_[idx_best, 0]),
            'covariance_type': self.params_score_[idx_best, 1]
        }
        self.best_estimator_ = GaussianMixture(**self.best_params_)\
            .fit(X)
        self.best_bic_ = self.params_score_[idx_best, 2]
        self.best_stats_ = {
            'means': self.all_means_[idx_best],
            'covariances': self.all_covariances_[idx_best]
        }

    def predict(self, X):
        """Predict labels using the base estimator

        Args:
            X (pandas DataFrame or numpy array): Data on which to predict

        Returns:
            np.ndarray: array with cluster predictions
        """
        return self.best_estimator_.predict(X)

    def export_best(self, out_path='.', out_name='best_gaussian_mixed'):
        """Export the best estimator as a pickle for persistence

        Args:
            out_path (str, optional): Path for output. Defaults to '.'.
            out_name (str, optional): Name for output. Defaults to
                'best_gaussian_mixed'.
        """
        out_path = out_path.rstrip('/')
        with open(f'{out_path}/{out_name}.pickle', 'wb') as f:
            pickle.dump(self.best_estimator_, f, pickle.HIGHEST_PROTOCOL)

    def _base_scatter(self):
        """
        n = self.best_params_['n_components']
        if n > 9:
            raise ValueError('Too many clusters for this method. '
                             'This method is for up to 9 clusters.'
                             'Use the data in the instance to build your own.')
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = self._colors[:n]
        marks = self._markers[:n]
        """
        pass

    def plot_best(
        self,
        X,
        show_fig=True,
        save_fig=False,
        out_path='.',
        out_name='best_GausMix'
    ):
        # check type of X
        if isinstance(X, pd.core.frame.DataFrame):
            predictions = self.predict(X)
            full = pd.concat([X, pd.Series(predictions)], axis=1)
            full.columns = list(X.columns) + ['cluster']
        elif isinstance(X, np.ndarray):
            predictions = self.predict(X)
            full = pd.concat(
                [pd.DataFrame(X), pd.Series(predictions)],
                axis=1
            )
            full.columns = list(full.columns)[:-1] + ['cluster']
        else:
            raise TypeError('Data should be either a pandas dataframe '
                            'or a numpy array')

        g = sns.pairplot(data=full, hue='cluster', height=3.5)
        n_c = self.best_params_['n_components']
        cov_t = self.best_params_['covariance_type']
        bic = float(self.best_bic_)
        g.fig.suptitle(
            f'Predicted clusters.\nNumber of Clusters: {n_c}    '
            f'Covariance Type: {cov_t}    '
            f'BIC: {bic:.2f}',
            y=1.08
        )
        if show_fig:
            plt.show()
        if save_fig:
            out_path = out_path.rstrip('/')
            g.savefig(f'{out_path}/{out_name}.png')
        plt.close()

    def plot_bic_comparison(
        self,
        show_fig=True,
        save_fig=False,
        out_path='.',
        out_name='GausMix_BIC_comparison'
    ):
        df = pd.DataFrame(
            self.params_score_,
            columns=['Number of Components', 'Covariance Type', 'BIC']
        )
        df['Number of Components'] = df['Number of Components'].astype(int)
        df.BIC = df.BIC.astype('float64')
        width = df['Number of Components'].nunique()
        fig, ax = plt.subplots(figsize=(3*width, 8))
        fig.suptitle('Comparison of BIC accros Num Components and Cov Type')
        sns.barplot(
            data=df, x='Number of Components', y='BIC',
            hue='Covariance Type', ax=ax
            )
        if show_fig:
            plt.show()
        if save_fig:
            out_path = out_path.rstrip('/')
            fig.savefig(f'{out_path}/{out_name}.png')
        plt.close()

    def plot_silhouette(
        self,
        X,
        show_fig=True,
        save_fig=False,
        out_path='.',
        out_name='best_GaussMix_silhouette'
    ):
        pass

    def plot_diagnostic(
        self,
        X,
        show_fig=True,
        save_fig=False,
        out_path='.',
        out_name='best_Diagnostic_plots'
    ):
        # check type of X
        if isinstance(X, pd.core.frame.DataFrame):
            predictions = self.predict(X)
            full = pd.concat([X, pd.Series(predictions)], axis=1)
            full.columns = list(X.columns) + ['cluster']
        elif isinstance(X, np.ndarray):
            predictions = self.predict(X)
            full = pd.concat(
                [pd.DataFrame(X), pd.Series(predictions)],
                axis=1
            )
            full.columns = list(full.columns)[:-1] + ['cluster']
        else:
            raise TypeError('Data should be either a pandas dataframe '
                            'or a numpy array')

        # create estimated vs empirical CDF
        # and qq plot against normal distribution
        n_features = X.shape[1]
        fig, ax = plt.subplots(n_features, 2, figsize=(12, 6*n_features))

        for i in range(n_features):
            pass

        if show_fig:
            plt.show()
        if save_fig:
            out_path = out_path.rstrip('/')
            fig.savefig(f'{out_path}/{out_name}.png')


# %%

if __name__ == '__main__':
    geyser = pd.read_csv('geyser.csv')

    # num_clust = np.arange(5)
    num_clust = 5
    # valores default:
    """
    mixmod = MixModSelection(estim=GaussianMixture(),
                             n_clust=np.arange(1, 10),
                             cov_type=['spherical', 'tied', 'diag', 'full'],
                             num_init=3
    )
    """
    # inicializo
    mixmod = MixModSelection(n_clust=num_clust)
    # ajusto
    mixmod.fit(geyser)
    # graficar el scatter del mejor ajuste
    mixmod.plot_best(geyser, save_fig=False)
    # graficar la comparación del bic para cada combinacion de
    # cluster y covariance type
    # mixmod.plot_bic_comparison(geyser, save_fig=True)
    # prediccion
    predictions = mixmod.predict(geyser)
    print('Mejores valores de parametros:', mixmod.best_params_)
    print('Mejor estimador:', mixmod.best_estimator_)
    print('Media del ajuste del mejor:', mixmod.best_stats_['means'])
    print('Covarianzas del ajuste del mejor:',
          mixmod.best_stats_['covariances'])

    # asi se guarda el estimador
    # mixmod.export_best()
# %%

# empirical cumulative distribution
"""
data = geyser.duration
data_sort = np.sort(data)
n = data.shape[0]
vals, counts = np.unique(data_sort, return_counts=True)
dens = counts/n
cumulative = np.cumsum(dens)
frequency = np.vstack((vals, cumulative)).T
print(frequency)
g = sns.lineplot(
    x=frequency[:, 0],
    y=frequency[:, 1],
    drawstyle='steps-post'
)
g.set(ylim=(-0.03, None))
g.axhline(0, linestyle='--')
g.axvline(x=frequency[0, 0], ymin=0.03, ymax=frequency[0, 1]/0.95)
plt.show()
plt.clf()
plt.show()
plt.clf()
# %%

one = np.array([1, 1, 2, 2, 2, 3, 4, 4, 5, 5])
n = len(one)
vals, counts = np.unique(one, return_counts=True)
dens = counts/n
cumulative = np.cumsum(dens)
frequency = np.vstack((vals, cumulative)).T
print(frequency)

g = sns.lineplot(
    x=frequency[:, 0],
    y=frequency[:, 1],
    drawstyle='steps-post'
)
g.set(ylim=(-0.03, None))
g.axhline(0, linestyle='--')
g.axvline(x=frequency[0, 0], ymin=0.03, ymax=frequency[0, 1]/0.95)
plt.show()
plt.clf()
"""

# %%
