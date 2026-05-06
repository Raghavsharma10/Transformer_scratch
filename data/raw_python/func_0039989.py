def getN21NG(rates, ver, lamb, br, reactfn):
    """
    excitation Franck-Condon factors (derived from Vallance Jones, 1974)
    """
    with h5py.File(str(reactfn), 'r', libver='latest') as f:
        A = f['/N2+1NG/A'].value
        lambdaA = f['/N2+1NG/lambda'].value.ravel(order='F')
        franckcondon = f['/N2+1NG/fc'].value

    return doBandTrapz(A, lambdaA, franckcondon, rates.loc[..., 'p1ng'], lamb, ver, rates.alt_km, br)