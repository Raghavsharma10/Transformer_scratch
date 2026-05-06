def getN21PG(rates, ver, lamb, br, reactfn):

    with h5py.File(str(reactfn), 'r', libver='latest') as fid:
        A = fid['/N2_1PG/A'].value
        lambnew = fid['/N2_1PG/lambda'].value.ravel(order='F')
        franckcondon = fid['/N2_1PG/fc'].value

    tau1PG = 1 / np.nansum(A, axis=1)
    """
    solve for base concentration
    confac=[1.66;1.56;1.31;1.07;.77;.5;.33;.17;.08;.04;.02;.004;.001];  %Cartwright, 1973b, stop at nuprime==12
    Gattinger and Vallance Jones 1974
    confac=array([1.66,1.86,1.57,1.07,.76,.45,.25,.14,.07,.03,.01,.004,.001])
    """

    consfac = franckcondon/franckcondon.sum()  # normalize
    losscoef = (consfac / tau1PG).sum()
    N01pg = rates.loc[..., 'p1pg'] / losscoef

    scalevec = (A * consfac[:, None]).ravel(order='F')  # for clarity (verified with matlab)

    vnew = scalevec[None, None, :] * N01pg.values[..., None]

    return catvl(rates.alt_km, ver, vnew, lamb, lambnew, br)