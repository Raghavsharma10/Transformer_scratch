def assert_equals(df1, df2, ignore_order=set(), ignore_indices=set(), all_close=False, _return_reason=False):
    '''
    Assert 2 data frames are equal

    A more verbose form of ``assert equals(df1, df2, ...)``. See `equals` for an explanation of the parameters.

    Parameters
    ----------
    df1 : ~pandas.DataFrame
        Actual data frame.
    df2 : ~pandas.DataFrame
        Expected data frame.
    ignore_order : ~typing.Set[int]
    ignore_indices : ~typing.Set[int]
    all_close : bool
    '''
    equals_, reason = equals(df1, df2, ignore_order, ignore_indices, all_close, _return_reason=True)
    assert equals_, '{}\n\n{}\n\n{}'.format(reason, df1.to_string(), df2.to_string())