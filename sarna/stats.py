import numpy as np
import scipy
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, levene
from borsar.stats import compute_regression_t
from borsar.utils import progressbar as progressbar_function


# TODO:
# - [ ] avoid calculating p, now it is computed but thrown away
#       (unnecessary computation time)
def ttest_ind_no_p(*args):
    t, p = stats.ttest_ind(*args)
    return t


def ttest_ind_welch_no_p(*args):
    t, p = stats.ttest_ind(*args, equal_var=False)
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


# - [ ] merge with corr?
# - [ ] add n_jobs to speed up?
def apply_stat(data, pred, y=None, stat_fun='OLS', interaction=None,
               center=True, progressbar=None):
    """
    Apply statistical test like ordinary least squares regression along
    the first dimension of the data.
    """
    import statsmodels.api as sm
    data_is_dep_var = y is None
    has_interaction = interaction is not None
    if has_interaction:
        test_pred = pred.copy()
        rand = np.random.normal(size=(test_pred.shape[0], 1))
        test_pred = np.append(test_pred, rand, axis=1)
        n_interactions = interaction(test_pred).shape[1]
    else:
        n_interactions = 0

    if data_is_dep_var:
        pred = sm.add_constant(pred)

    if stat_fun == 'OLS':
        def stat_fun(dt, pred):
            mdl = sm.OLS(dt, pred).fit(disp=False)
            return mdl.tvalues, mdl.pvalues
    elif stat_fun == 'logistic':
        def stat_fun(dt, pred):
            mdl = sm.Logit(dt, pred).fit(disp=False)
            return mdl.tvalues, mdl.pvalues

    orig_data_shape = list(data.shape)
    if data.ndim > 2:
        data = data.reshape([orig_data_shape[0], np.prod(orig_data_shape[1:])])

    # FIXME - scale and center; do the same to predictors
    if center and not data_is_dep_var:
        data = ((data - data.mean(axis=0, keepdims=True)) /
                 data.std(axis=0, keepdims=True))

    # check dims and allocate output
    n_preds, n_comps = pred.shape[1], data.shape[1]
    n_preds += n_interactions + int(not data_is_dep_var)
    tvals = np.zeros((n_preds, n_comps))
    pvals = np.zeros((n_preds, n_comps))

    pbar = progressbar_function(progressbar, total=n_comps)

    # perform model for each
    for idx in range(n_comps):
        if not data_is_dep_var:
            data_pred = data[:, [idx]]
            data_pred = np.concatenate([pred, data_pred], axis=1)

            if interaction:
                data_pred = np.concatenate([data_pred, interaction(data_pred)],
                                           axis=1)

            # run model
            tval, pval = stat_fun(y, data_pred)
        else:
            # FIXME - if OLS -> use borsar.stats.compute_regression_t
            tval, pval = stat_fun(data, pred)
        tvals[:, idx] = tval
        pvals[:, idx] = pval

        pbar.update(1)

    new_shp = [n_preds] + orig_data_shape[1:]
    tvals = tvals.reshape(new_shp)
    pvals = pvals.reshape(new_shp)
    pbar.close()
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


def iterative_cleaning(data, sd_threshold=3, min_retain=10, verbose=True):
    n_tri = data.shape[0]
    retain = np.ones(n_tri, dtype=bool)

    got_bad = True
    n_good = n_tri

    step_idx = 0
    while got_bad and n_good > min_retain:
        step_idx += 1
        current_idx = np.where(retain)[0]
        data_z = stats.zscore(data[current_idx], axis=0)
        bad = data_z > sd_threshold

        got_bad = bad.any()
        if got_bad:
            retain[current_idx[bad]] = False
            n_good = retain.sum()
            if verbose:
                print(f'Step {step_idx}: removed {bad.sum()} observations.')
        elif verbose:
            print(f'Step {step_idx}: no more to remove.\nRemoved '
                    f'{(~retain).sum()} observations in total.')

    return retain