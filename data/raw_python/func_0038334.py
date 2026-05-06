def pprint(data, names='', title='', formats={}):
    """
    Prints tables with a bit of formatting

    Parameters
    ----------
    data: (sequence, dict, table)
        The data to print in the table
    names: sequence
        The column names
    title: str (optional)
        The title of the table
    formats: dict
        A dictionary of column:format values

    """
    # Make the data into a table if it isn't already
    if type(data) != at.Table:
        data = at.Table(data, names=names)

    # Make a copy
    pdata = data.copy()

    # Put the title in the metadata
    try:
        title = title or pdata.meta['name']
    except:
        pass

    # Shorten the column names for slimmer data
    for old, new in zip(*[pdata.colnames, [
        i.replace('wavelength', 'wav').replace('publication', 'pub').replace('instrument', 'inst')\
        .replace('telescope','scope') for i in pdata.colnames]]):
        pdata.rename_column(old, new) if new != old else None

    # Format the columns
    formats.update({'comments': '%.15s', 'obs_date': '%.10s', 'names': '%.30s', 'description': '%.50s'})

    # print it!
    if title: print('\n' + title)
    try:
        ii.write(pdata, sys.stdout, Writer=ii.FixedWidthTwoLine, formats=formats, fill_values=[('None', '-')])
    except UnicodeDecodeError:  # Fix for Unicode characters. Print out in close approximation to ii.write()
        max_length = 50
        str_lengths = dict()
        for key in pdata.keys():
            lengths = map(lambda x: len(str(x).decode('utf-8')), pdata[key].data)
            lengths.append(len(key))
            str_lengths[key] = min(max(lengths), max_length)
        print(' '.join(key.rjust(str_lengths[key]) for key in pdata.keys()))
        print(' '.join('-' * str_lengths[key] for key in pdata.keys()))
        for i in pdata:
            print(' '.join([str(i[key]).decode('utf-8')[:max_length].rjust(str_lengths[key])
                           if i[key] else '-'.rjust(str_lengths[key]) for key in pdata.keys()]))