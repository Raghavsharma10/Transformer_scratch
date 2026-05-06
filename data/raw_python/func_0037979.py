def dict_tovot(tabdata, tabname='votable.xml', phot=False, binary=True):
    """
    Converts dictionary table **tabdata** to a VOTable with name **tabname**

    Parameters
    ----------
    tabdata: list
      SQL query dictionary list from running query_dict.execute()
    tabname: str
      The name of the VOTable to be created
    phot: bool
      Parameter specifying if the table contains photometry to be merged
    binary: bool
      Parameter specifying if the VOTable should be saved as a binary.
      This is necessary for tables with lots of text columns.

    """

    # Check if input is a dictionary
    if not isinstance(tabdata[0], dict):
        raise TypeError('Table must be a dictionary. Call the SQL query with query_dict.execute()')

    # Create an empty table to store the data
    t = Table()

    colnames = tabdata[0].keys()

    # If this is a photometry table, parse it and make sure to have the full list of columns
    if phot:
        tabdata = photparse(tabdata)

        colnames = tabdata[0].keys()

        for i in range(len(tabdata)):
            tmpcol = tabdata[i].keys()
            for elem in tmpcol:
                if elem not in colnames:
                    colnames.append(elem)

        # No need for band column any more
        try:
            colnames.remove('band')
        except ValueError:
            pass

    # Run through all the columns and create them
    for elem in colnames:
        table_add(t, tabdata, elem)

    # Output to a file
    print('Creating table...')
    votable = from_table(t)

    # Required in some cases (ie, for lots of text columns)
    if binary:
        votable.set_all_tables_format('binary')

    votable.to_xml(tabname)

    print('Table created: {}'.format(tabname))