def stack_eggs(eggs, meta='concatenate'):
    '''
    Takes a list of eggs, stacks them and reindexes the subject number

    Parameters
    ----------
    eggs : list of Egg data objects
        A list of Eggs that you want to combine
    meta : string
        Determines how the meta data of each Egg combines. Default is 'concatenate'
        'concatenate' concatenates keys in meta data dictionary shared between eggs, and copies non-overlapping keys
        'separate' keeps the Eggs' meta data dictionaries separate, with each as a list index in the stacked meta data


    Returns
    ----------
    new_egg : Egg data object
        A mega egg comprised of the input eggs stacked together

    '''
    from .egg import Egg

    pres = [egg.pres.loc[sub,:].values.tolist() for egg in eggs for sub in egg.pres.index.levels[0].values.tolist()]
    rec = [egg.rec.loc[sub,:].values.tolist() for egg in eggs for sub in egg.rec.index.levels[0].values.tolist()]

    if meta is 'concatenate':
        new_meta = {}
        for egg in eggs:
            for key in egg.meta:
                if key in new_meta:
                    new_meta[key] = list(new_meta[key])
                    new_meta[key].extend(egg.meta.get(key))
                else:
                    new_meta[key] = egg.meta.get(key)

    elif meta is 'separate':
        new_meta = list(egg.meta for egg in eggs)

    return Egg(pres=pres, rec=rec, meta=new_meta)