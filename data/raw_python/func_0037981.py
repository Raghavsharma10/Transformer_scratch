def photparse(tab):
    """
    Parse through a photometry table to group by source_id

    Parameters
    ----------
    tab: list
      SQL query dictionary list from running query_dict.execute()

    Returns
    -------
    newtab: list
      Dictionary list after parsing to group together sources

    """

    # Check that source_id column is present
    if 'source_id' not in tab[0].keys():
        raise KeyError('phot=TRUE requires the source_id columb be included')

    # Loop through the table and grab unique band names and source IDs
    uniqueid = []
    for i in range(len(tab)):
        tmpid = tab[i]['source_id']

        if tmpid not in uniqueid:
            uniqueid.append(tmpid)

    # Loop over unique id and create a new table for each element in it
    newtab = []
    for sourceid in uniqueid:
        tmpdict = photaddline(tab, sourceid)
        newtab.append(tmpdict)

    return newtab