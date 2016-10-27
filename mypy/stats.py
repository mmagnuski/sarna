import numpy as np
from scipy.stats import ttest_ind, ttest_rel


# TODO: compute t's without p (less computation time)
def ttest_ind_no_p(*args):
    t, p = ttest_ind(*args)
    return t


def ttest_rel_no_p(*args):
    t, p = ttest_rel(*args)
    return t


# - [ ] pred -> data; data + pred -> y
# - [ ] progressbar
# - [ ] apply model - statsmodels or sklearn
def apply_regr(data, pred, along=0):
    """
    apply ordinary least squares regression along specified
    dimension of the data.
    """
    import statsmodels.api as sm
    pred = sm.add_constant(pred)

    # reshape data to ease up regression
    if not along == 0:
        dims = list(range(data.ndim))
        dims.remove(along)
        dims = [along] + dims
        data = np.transpose(data, dims)
    shp = list(data.shape)
    if data.ndim > 2:
        data = data.reshape([shp[0], np.prod(shp[1:])])
    data = data.T

    # check dims and allocate output
    n_comps = data.shape[0]
    n_preds = pred.shape[1]
    tvals = np.zeros((n_comps, n_preds))
    pvals = np.zeros((n_comps, n_preds))

    # perform refression for each
    for ii, dt in enumerate(data):
        mdl = sm.OLS(dt, pred).fit()
        tvals[ii, :] = mdl.tvalues
        pvals[ii, :] = mdl.pvalues
    new_shp = [n_preds] + shp[1:]

    tvals = tvals.T.reshape(new_shp)
    pvals = pvals.T.reshape(new_shp)

    return tvals, pvals


# TODO:
# - [ ] should the default be along -1?
# - [ ] check whether n_preds is there in inverse_transform
class Reshaper(object):
    def __init__(self):
        self.shape = None
        self.along = None
        self.todims = False

    def fit(self, X, along=-1):
        # reshape data to ease up regression
        if along == -1:
            along = X.ndim - 1
        if not along == 0:
            dims = list(range(data.ndim))
            dims.remove(along)
            self.todims = [along] + dims
        self.along = along
        self.shape = list(data.shape)

    def transform(self, X):
        if not self.along == 0:
            X = np.transpose(X, self.todims)
        if X.ndim > 2:
            this_shape = X.shape
            X = X.reshape([this_shape[0], np.prod(this_shape[1:])])
        return X.T # may not be necessary if no diff in performance

    def inverse_transform(self, X):
        n_preds = X.shape[-1]
        new_shp = [n_preds] + self.shape[1:]
        return X.T.reshape(new_shp)


# goodness of fit
def log_likelihood(data, distrib, params=None, binomial=False):
    if params is None:
        params = distrib.fit(data)
    if not binomial:
        return np.sum(np.log(distrib.pdf(data, *params)))
    else:
        prediction = distrib.pdf(data, *params)
        return np.sum(np.log(prediction) * data +
                      np.log(1 - prediction) * (1 - data))
