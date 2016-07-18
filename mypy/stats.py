import numpy as np
import statsmodels.api as sm


# - [ ] pred -> data; data + pred -> y
# - [ ] progressbar
# - [ ] apply model - statsmodels or sklearn
def apply_regr(data, pred, along=0):
    """
    apply ordinary least squares regression along specified
    dimension of the data.
    """
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
