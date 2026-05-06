def chk_col_numbers(line_num, num_cols, tax_id_col, id_col, symbol_col):
    """
    Check that none of the input column numbers is out of range.
    (Instead of defining this function, we could depend on Python's built-in
    IndexError exception for this issue, but the IndexError exception wouldn't
    include line number information, which is helpful for users to find exactly
    which line is the culprit.)
    """

    bad_col = ''
    if tax_id_col >= num_cols:
        bad_col = 'tax_id_col'
    elif id_col >= num_cols:
        bad_col = 'discontinued_id_col'
    elif symbol_col >= num_cols:
        bad_col = 'discontinued_symbol_col'

    if bad_col:
        raise Exception(
            'Input file line #%d: column number of %s is out of range' %
            (line_num, bad_col))