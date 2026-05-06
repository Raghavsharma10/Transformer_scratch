def _strip_column_name(col_name, keep_paren_contents=True):
    """
    Utility script applying several regexs to a string.
    Intended to be used by `strip_column_names`.

    This function will:
        1. replace informative punctuation components with text
        2. (optionally) remove text within parentheses
        3. replace remaining punctuation/whitespace with _
        4. strip leading/trailing punctuation/whitespace

    Parameters
    ----------
    col_name (str): input character string
    keep_paren_contents (logical):
        controls behavior of within-paren elements of text
         - if True, (the default) all text within parens retained
         - if False, text within parens will be removed from the field name

    Returns
    --------
    modified string for new field name

    Examples
    --------
    > print([_strip_column_name(col) for col in ['PD-L1','PD L1','PD L1_']])
    """
    # start with input
    new_col_name = col_name
    # replace meaningful punctuation with text equivalents
    # surround each with whitespace to enforce consistent use of _
    punctuation_to_text = {
        '<=': 'le',
        '>=': 'ge',
        '=<': 'le',
        '=>': 'ge',
        '<': 'lt',
        '>': 'gt',
        '#': 'num'
    }
    for punctuation, punctuation_text in punctuation_to_text.items():
        new_col_name = new_col_name.replace(punctuation, punctuation_text)

    # remove contents within ()
    if not(keep_paren_contents):
        new_col_name = re.sub('\([^)]*\)', '', new_col_name)

    # replace remaining punctuation/whitespace with _
    punct_pattern = '[\W_]+'
    punct_replacement = '_'
    new_col_name = re.sub(punct_pattern, punct_replacement, new_col_name)

    # remove leading/trailing _ if it exists (if last char was punctuation)
    new_col_name = new_col_name.strip("_")

    # TODO: check for empty string

    # return lower-case version of column name
    return new_col_name.lower()