def get_ldict_keys(ldict, flatten_keys=False, **kwargs):
    """
    Get first level keys from a list of dicts
    """
    result = []
    for ddict in ldict:
        if isinstance(ddict, dict):

            if flatten_keys:
                ddict = flatten(ddict, **kwargs)

            result.extend(ddict.keys())
    return list(set(result))