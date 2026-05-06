def photaddline(tab, sourceid):
    """
    Loop through the dictionary list **tab** creating a line for the source specified in **sourceid**

    Parameters
    ----------
    tab:
      Dictionary list of all the photometry data
    sourceid:
      ID of source in the photometry table (source_id)

    Returns
    -------
    tmpdict: dict
        Dictionary with all the data for the specified source

    """

    colnames = tab[0].keys()
    tmpdict = dict()
    for i in range(len(tab)):

        # If not working on the same source, continue
        if tab[i]['source_id'] != sourceid:
            continue

        # Check column names and create new ones for band-specific ones
        for elem in colnames:
            if elem not in ['comments', 'epoch', 'instrument_id', 'magnitude', 'magnitude_unc', 'publication_id',
                            'system', 'telescope_id']:
                tmpdict[elem] = tab[i][elem]
            elif elem == 'band':
                continue
            else:
                tmpstr = tab[i]['band']+'.'+elem
                tmpdict[tmpstr] = tab[i][elem]

    return tmpdict