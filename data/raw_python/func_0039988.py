def getAtomic(rates, ver, lamb, br, reactfn):
    """ prompt atomic emissions (nm)
    844.6 777.4
    """
    with h5py.File(reactfn, 'r') as f:
        lambnew = f['/atomic/lambda'].value.ravel(order='F')  # some are not 1-D!

    vnew = np.concatenate((rates.loc[..., 'po3p3p'].values[..., None],
                           rates.loc[..., 'po3p5p'].values[..., None]), axis=-1)

    return catvl(rates.alt_km, ver, vnew, lamb, lambnew, br)