def remove_dependent_cols(M, tol=1e-6, display=False):
    """
    Returns a matrix where dependent columsn have been removed
    """
    R = la.qr(M, mode='r')[0][:M.shape[1], :]
    I = (abs(R.diagonal())>tol)
    if sp.any(~I) and display:
        print(('cols ' + str(sp.where(~I)[0]) +
                ' have been removed because linearly dependent on the others'))
        R = M[:,I]
    else:
        R = M.copy()
    return R