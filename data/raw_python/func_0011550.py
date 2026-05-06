def _validate_row(row, sep=',', required_length=None):
    '''validate_row will ensure that a row has the proper length, and is
       not empty and cleaned of extra spaces.
 
       Parameters
       ==========
       row: a single row, not yet parsed.

       Returns a valid row, or None if not valid

    '''
    if not isinstance(row, list):
        row = _parse_row(row, sep)

    if required_length:
        length = len(row)
        if length != required_length:
            bot.warning('Row should have length %s (not %s)' %(required_length,
                                                               length))
            bot.warning(row) 
            row = None

    return row