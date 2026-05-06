def clean_dictkeys(ddict, exclusions=None):
    """
    Exclude chars in dict keys and return a clean dictionary.
    """
    exclusions = [] if exclusions is None else exclusions

    if not isinstance(ddict, dict):
        return {}

    for key in list(ddict.keys()):
        if [incl for incl in exclusions if incl in key]:
            data = ddict.pop(key)
            clean_key = exclude_chars(key, exclusions)

            if clean_key:
                if clean_key in ddict:
                    ddict[clean_key] = force_list(ddict[clean_key])
                    add_element(ddict, clean_key, data)
                else:
                    ddict[clean_key] = data

        # dict case
        if isinstance(ddict.get(key), dict):
            ddict[key] = clean_dictkeys(ddict[key], exclusions)

        # list case
        elif isinstance(ddict.get(key), list):
            for row in ddict[key]:
                if isinstance(row, dict):
                    row = clean_dictkeys(row, exclusions)

    return ddict