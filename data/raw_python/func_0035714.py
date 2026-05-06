def assert_equals(actual, expected, ignore_order=False, ignore_index=False, all_close=False):
    '''
    Assert 2 series are equal.

    Like ``assert equals(series1, series2, ...)``, but with better hints at
    where the series differ. See `equals` for
    detailed parameter doc.

    Parameters
    ----------
    actual : ~pandas.Series
    expected : ~pandas.Series
    ignore_order : bool
    ignore_index : bool
    all_close : bool
    '''
    equals_, reason = equals(actual, expected, ignore_order, ignore_index, all_close, _return_reason=True)
    assert equals_, '{}\n\n{}\n\n{}'.format(reason, actual.to_string(), expected.to_string())