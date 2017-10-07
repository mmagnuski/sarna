import numpy as np
import scipy
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, levene


# TODO:
# - [ ] avoid calculating p, now it is computed but thrown away
#       (unnecessary computation time)
def ttest_ind_no_p(*args):
    t, p = stats.ttest_ind(*args)
    return t


def ttest_rel_no_p(*args):
    t, p = stats.ttest_rel(*args)
    return t


# TODO:
# - [ ] seems that y has to be a vector now, adapt for matrix - matrix
def corr(x, y, method='Pearson'):
    '''correlate two vectors/matrices.

    This function can be useful because scipy.stats.pearsonr does too little
    (takes only vectors) and scipy.stats.spearmanr does too much (calculates
    all possible correlations when given two matrices - instead of correlating
    only pairs of variables where one is from the first and the other from  the
    second matrix)
    '''
    x_size = x.shape
    y_size = y.shape
    if len(x_size) == 1:
        x = x[:, np.newaxis]
        x_size = x.shape

    if method == 'Pearson':
        from scipy.stats import pearsonr as cor
    elif method == 'Spearman':
        from scipy.stats import spearmanr as cor

    rs = list()
    ps = list()
    for col in range(x_size[1]):
        r, p = cor(x[:, col], y)
        rs.append(r)
        ps.append(p)
    return np.array(rs), np.array(ps)


# - [ ] merge with apply_test (and corr?)
# - [ ] pred -> data; data + pred -> y
# - [ ] progressbar
# - [ ] use faster function for ols (a wrapper around np.linalg.lstsq)
# - [ ] apply model - statsmodels (or sklearn?)
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

    # perform regression for each (using np.linalg.lstsq would be faster)
    for ii, dt in enumerate(data):
        mdl = sm.OLS(dt, pred).fit()
        tvals[ii, :] = mdl.tvalues
        pvals[ii, :] = mdl.pvalues
    new_shp = [n_preds] + shp[1:]

    tvals = tvals.T.reshape(new_shp)
    pvals = pvals.T.reshape(new_shp)

    return tvals, pvals


def apply_test(data, group, test):
    '''applies test along axis=1
    data - 2d data array
    group - group identity (rows)
    test - 'levene' for example
            should accept functions too
    '''
    n_samples = data.shape[1]
    if test == 'levene':
        levene_W = np.zeros(n_samples)
        levene_p = np.zeros(n_samples)
        for t_ind in range(n_samples):
            levene_W[t_ind], levene_p[t_ind] = stats.levene(
                data[group == 0, t_ind], data[group == 1, t_ind])

        return levene_W, levene_p
    else:
        raise NotImplementedError('Only levene is implemented currently...')


# - [ ] compute t for all preds when multiple regressors
# - [ ] consider using in apply_test or apply_regr (then would need p values)
# - [x] compare speed with respect to using statsmodels
#   (scipy loop is about 2 times faster than statsmodels)
# - [x] compare speed with using scipy.linalg.lstsq(a, b) where b is 2d
#   (scipy noloop is about 150 times faster than scipy loop)
def compute_regression_t(data, preds, backend='scipy'):
    '''Compute regression t values for whole 2d data space.

    Parameters
    ----------
    data : numpy array of shape (observations, dim1, dim2)
        Data to perform regression on. Frist dimension are observations
        (for example trials or subjects). Has to be 3d unless using scipy
        backend.
    preds : numpy array
        Predictor array. Currently only 1d array is supported.
    '''
    if backend == 'scipy':
        n_obs, _ = data.shape
    else:
        n_obs, n_channels, n_times = data.shape
    assert preds.ndim == 1 or preds.shape[1] == 1, ('currently only single'
        ' predictor is allowed (constant term is added automatically).')
    assert n_obs == preds.shape[0], ('preds must have the same number of rows'
        ' as the size of first data dimension (observations).')

    # add constant term
    if preds.ndim == 1: preds = np.atleast_2d(preds).T
    preds = np.concatenate([np.ones((n_obs, 1)), preds], axis=1)

    if backend == 'scipy_loop':
        t_vals = np.zeros((n_channels, n_times))
        mean_x = preds[:, 1].mean()
        x_sd = np.sqrt(np.sum((preds[:, 1] - mean_x) ** 2))
        for ch in range(n_channels):
            for tm in range(n_times):
                coefs, _, _, _ = scipy.linalg.lstsq(preds, data[:, ch, tm])
                prediction = (preds * coefs[np.newaxis, :]).sum(axis=1)
                squared_error = (data[:, ch, tm] - prediction) ** 2
                prediction_error = np.sqrt(np.sum(squared_error) / (n_obs - 2))

                SE = prediction_error / x_sd
                t_vals[ch, tm] = coefs[1] / SE

    elif backend == 'scipy':
        mean_x = preds[:, 1].mean()
        x_sd = np.sqrt(np.sum((preds[:, 1] - mean_x) ** 2))

        original_shape = data.shape
        data = data.reshape((original_shape[0], np.prod(original_shape[1:])))
        coefs, _, _, _ = scipy.linalg.lstsq(preds, data)

        prediction = (preds[:, :, np.newaxis] * coefs[np.newaxis, :]
                      ).sum(axis=1)
        squared_error = (data - prediction) ** 2
        prediction_error = np.sqrt(np.sum(squared_error, axis=0) / (n_obs - 2))
        SE = prediction_error / x_sd
        t_vals = (coefs[1, :] / SE).reshape(original_shape[1:])

    elif backend == 'statsmodels':
        t_vals = np.zeros((n_channels, n_times))
        for ch in range(n_channels):
            for tm in range(n_times):
                mdl = sm.OLS(data[:, ch, tm], preds).fit()
                t_vals[ch, tm] = mdl.tvalues[1]
    return t_vals


# TODO:
# - [ ] rename along to retain
# - [ ] should the default be along -1?
# - [ ] add fit_transform
# - [ ] check whether n_preds is there in inverse_transform
# - [ ] use rollaxis if not along default
# - [ ] allow more than one dim in retain?
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


def format_pvalue(pvalue):
    if pvalue > .001:
        return 'p = {:.3f}'.format(pvalue)
    else:
        powers = 10 ** np.arange(-3, -101, -1, dtype='float')
        which_power = np.where(pvalue < powers)[0][-1]
        if which_power < 2:
            return 'p < {}'.format(['0.001', '0.0001'][which_power])
        else:
            return 'p < {}'.format(str(powers[which_power]))


def confidence_interval(arr, ci):
    """Calculate the `ci` parametric confidence interval for array `arr`.
    Computes the ci from t distribution with relevant mean and scale.
    """
    from scipy import stats
    mean, sigma = arr.mean(axis=0), stats.sem(arr, axis=0)
    return stats.t.interval(ci, loc=mean, scale=sigma, df=arr.shape[0])
