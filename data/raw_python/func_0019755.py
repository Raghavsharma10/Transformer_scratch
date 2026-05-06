def shuffle_egg(egg):
    """ Shuffle an Egg's recalls"""
    from .egg import Egg
    pres, rec, features, dist_funcs = parse_egg(egg)

    if pres.ndim==1:
        pres = pres.reshape(1, pres.shape[0])
        rec = rec.reshape(1, rec.shape[0])
        features = features.reshape(1, features.shape[0])

    for ilist in range(rec.shape[0]):
        idx = np.random.permutation(rec.shape[1])
        rec[ilist,:] = rec[ilist,idx]

    return Egg(pres=pres, rec=rec, features=features, dist_funcs=dist_funcs)