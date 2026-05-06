def getMetastable(rates, ver: np.ndarray, lamb, br, reactfn: Path):
    with h5py.File(reactfn, 'r') as f:
        A = f['/metastable/A'][:]
        lambnew = f['/metastable/lambda'].value.ravel(order='F')  # some are not 1-D!

    """
    concatenate along the reaction dimension, axis=-1
    """
    vnew = np.concatenate((A[:2] * rates.loc[..., 'no1s'].values[:, None],
                           A[2:4] * rates.loc[..., 'no1d'].values[:, None],
                           A[4:] * rates.loc[..., 'noii2p'].values[:, None]), axis=-1)

    assert vnew.shape == (rates.shape[0], A.size)

    return catvl(rates.alt_km, ver, vnew, lamb, lambnew, br)