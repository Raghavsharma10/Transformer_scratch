def _read_runtime_vars(variable_file, sep=','):
    '''read the entire runtime variable file, and return a list of lists,
       each corresponding to a row. We also check the header, and exit
       if anything is missing or malformed.

       Parameters
       ==========

       variable_file: full path to the tabular file with token, exp_id, etc.
       sep: the default delimiter to use, if not set in enironment.

       Returns
       =======

       valid_rows: a list of lists, each a valid row

           [['test-parse-url', 'globalname', 'globalvalue', '*'],
            ['test-parse-url', 'color', 'red', '123'], 
            ['test-parse-url', 'color', 'blue', '456'],
            ['test-parse-url', 'words', 'at the thing', '123'],
            ['test-parse-url', 'words', 'omg tacos', '456']]

    '''

    rows = [x for x in read_file(variable_file).split('\n') if x.strip()]
    valid_rows = []

    if len(rows) > 0:

        # Validate header and rows, exit if not valid

        header = rows.pop(0).split(sep)
        validate_header(header)
        for row in rows:
            row = _validate_row(row, sep=sep, required_length=4)

            # If the row is returned, it is valid

            if row:
                valid_rows.append(row)

    return valid_rows