def strip_column_names(cols, keep_paren_contents=True):
    """
    Utility script for renaming pandas columns to patsy-friendly names.

    Revised names have been:
        - stripped of all punctuation and whitespace (converted to text or `_`)
        - converted to lower case

    Takes a list of column names, returns a dict mapping
    names to revised names.

    If there are any concerns with the conversion, this will
    print a warning & return original column names.

    Parameters
    ----------

    cols (list): list of strings containing column names
    keep_paren_contents (logical):
        controls behavior of within-paren elements of text
         - if True, (the default) all text within parens retained
         - if False, text within parens will be removed from the field name

    Returns
    -------

    dict mapping col_names -> new_col_names

    Example
    -------

    > df = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
      'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd']),
      'PD L1 (value)': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd']),
      'PD L1 (>1)': pd.Series([0., 1., 1., 0.], index=['a', 'b', 'c', 'd']),
      }
    > df = pd.DataFrame(df)
    > df = df.rename(columns = strip_column_names(df.columns))

    ## observe, by comparison
    > df2 = df.rename(columns = strip_column_names(df.columns,
        keep_paren_contents=False))
    """

    # strip/replace punctuation
    new_cols = [
        _strip_column_name(col, keep_paren_contents=keep_paren_contents)
        for col in cols]

    if len(new_cols) != len(set(new_cols)):
        warn_str = 'Warning: strip_column_names (if run) would introduce duplicate names.'
        warn_str += ' Reverting column names to the original.'

        warnings.warn(warn_str, Warning)
        print('Warning: strip_column_names would introduce duplicate names. Please fix & try again.')
        return dict(zip(cols, cols))

    return dict(zip(cols, new_cols))