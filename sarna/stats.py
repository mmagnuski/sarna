import numpy as np
import scipy
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, levene
from borsar.stats import compute_regression_t

from .utils import progressbar as progressbar_function


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
# - [x] seems that y has to be a vector now, adapt for matrix - matrix
#       (done for Pearson)
# - [ ] look for better implementations
def corr(x, y, method='Pearson'):
    '''correlate two vectors/matrices.

    This function can be useful because scipy.stats.pearsonr does too little
    (takes only vectors) and scipy.stats.spearmanr does too much (calculates
    all possible correlations when given two matrices - instead of correlating
    only pairs of variables where one is from the first and the other from  the
    second matrix)
    '''
    if x.ndim == 1:
        x = x[:, np.newaxis]
        x_size = x.shape

    if method == 'Pearson':
        from scipy.stats import pearsonr as cor
    elif method == 'Spearman':
        from scipy.stats import spearmanr as cor

    rs = list()
    ps = list()
    if method == 'Spearman':
        for col in range(x.shape[1]):
            r, p = cor(x[:, col], y)
            rs.append(r)
            ps.append(p)
        return np.array(rs), np.array(ps)
    else:
        rmat = np.zeros((x.shape[1], y.shape[1]))
        pmat = rmat.copy()
        for x_idx in range(x.shape[1]):
            for y_idx in range(y.shape[1]):
                r, p = cor(x[:, x_idx], y[:, y_idx])
                rmat[x_idx, y_idx] = r
                pmat[x_idx, y_idx] = p
        return rmat, pmat


# - [x] two modes of prediction: pred -> data; data + covariates -> pred
# - [ ] consider renaming: data would remain as the multidimensional data
#       pred would be predictors
#       and y would be None - that is if y is other than `data`
#       no need to awkwardly swap the variables then too!
# - [x] apply model - statsmodels (might be even sklearn)
# - [ ] merge with apply_test (and corr?)
# - [ ] add n_jobs to speed up
def apply_stat(data, pred, along=0, stat_fun='OLS', covariates=None,
               interaction=None, center=True, progressbar=None):
    """
    Apply statistical test like ordinary least squares regression along
    specified dimension of the data.
    """
    import statsmodels.api as sm
    has_covar = covariates is not None
    has_interaction = interaction is not None

    # do we want to also add constat to the covariates?
    if not has_covar:
        pred = sm.add_constant(pred)

    if stat_fun == 'OLS':
        def stat_fun(dt, pred):
            mdl = sm.OLS(dt, pred).fit(disp=False)
            return mdl.tvalues, mdl.pvalues
    elif stat_fun == 'logistic':
        def stat_fun(dt, pred):
            mdl = sm.Logit(dt, pred).fit(disp=False)
            return mdl.tvalues, mdl.pvalues

    if has_covar:
        data, pred = pred, data

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
    add_dim = 1 if has_interaction else 0
    add_dim = add_dim + covariates.shape[1] if has_covar else add_dim
    n_comps = data.shape[0]
    n_preds = pred.shape[1] + add_dim
    tvals = np.zeros((n_comps, n_preds))
    pvals = np.zeros((n_comps, n_preds))

    pbar = progressbar_function(progressbar, total=n_comps)

    # perform model for each
    for ii, dt in enumerate(data):
        if covariates is not None:
            prd = dt[:, np.newaxis]

            # center given predictor
            # TODO - change to center / scale args
            #        also do that at the beginning to speed up
            if center:
                prd -= prd.mean()
                prd /= prd.std()
            prd = np.concatenate([covariates, prd], axis=1)

            if interaction:
                add_interaction = interaction(prd)
                prd = np.concatenate([prd, add_interaction], axis=1)

            # run model (we use `pred` as y because data and pred were swapped)
            tval, pval = stat_fun(pred, prd)
        else:
            tval, pval = stat_fun(dt, pred)
        tvals[ii, :] = tval
        pvals[ii, :] = pval

        pbar.update(1)

    new_shp = [n_preds] + shp[1:]
    tvals = tvals.T.reshape(new_shp)
    pvals = pvals.T.reshape(new_shp)

    return tvals, pvals


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


def confidence_interval(arr, ci):
    """Calculate the `ci` parametric confidence interval for array `arr`.
    Computes the ci from t distribution with relevant mean and scale.
    """
    from scipy import stats
    mean, sigma = arr.mean(axis=0), stats.sem(arr, axis=0)
    return stats.t.interval(ci, loc=mean, scale=sigma, df=arr.shape[0])
