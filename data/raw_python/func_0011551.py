def _parse_row(row, sep=','):
    '''parse row is a helper function to simply clean up a string, and parse
       into a row based on a delimiter. If a required length is provided,
       we check for this too.

    '''
    parsed = row.split(sep)
    parsed = [x for x in parsed if x.strip()]
    return parsed