def crack_egg(egg, subjects=None, lists=None):
    '''
    Takes an egg and returns a subset of the subjects or lists

    Parameters
    ----------
    egg : Egg data object
        Egg that you want to crack

    subjects : list
        List of subject idxs

    lists : list
        List of lists idxs

    Returns
    ----------
    new_egg : Egg data object
        A sliced egg, good on a salad

    '''
    from .egg import Egg

    if hasattr(egg, 'features'):
        all_have_features = egg.features is not None
    else:
        all_have_features=False
    opts = {}

    if subjects is None:
        subjects = egg.pres.index.levels[0].values.tolist()
    elif type(subjects) is not list:
        subjects = [subjects]

    if lists is None:
        lists = egg.pres.index.levels[1].values.tolist()
    elif type(lists) is not list:
        lists = [lists]

    idx = pd.IndexSlice
    pres = egg.pres.loc[idx[subjects,lists],egg.pres.columns]
    rec = egg.rec.loc[idx[subjects,lists],egg.rec.columns]

    pres = [pres.loc[sub,:].values.tolist() for sub in subjects]
    rec = [rec.loc[sub,:].values.tolist() for sub in subjects]

    if all_have_features:
        features = egg.features.loc[idx[subjects,lists],egg.features.columns]
        opts['features'] = [features.loc[sub,:].values.tolist() for sub in subjects]

    return Egg(pres=pres, rec=rec, **opts)