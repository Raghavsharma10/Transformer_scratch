def align_data(*args):
    """
    Resample and aligh data for defined frequency

    Args:
        *args: DataFrame of data to be aligned

    Returns:
        pd.DataFrame: aligned data with renamed columns

    Examples:
        >>> start = '2018-09-10T10:10:00'
        >>> tz = 'Australia/Sydney'
        >>> idx = pd.date_range(start=start, periods=6, freq='min').tz_localize(tz)
        >>> close_1 = [31.08, 31.10, 31.11, 31.07, 31.04, 31.04]
        >>> vol_1 = [10166, 69981, 14343, 10096, 11506, 9718]
        >>> d1 = pd.DataFrame(dict(price=close_1, volume=vol_1), index=idx)
        >>> d1
                                   price  volume
        2018-09-10 10:10:00+10:00  31.08   10166
        2018-09-10 10:11:00+10:00  31.10   69981
        2018-09-10 10:12:00+10:00  31.11   14343
        2018-09-10 10:13:00+10:00  31.07   10096
        2018-09-10 10:14:00+10:00  31.04   11506
        2018-09-10 10:15:00+10:00  31.04    9718
        >>> close_2 = [70.81, 70.78, 70.85, 70.79, 70.79, 70.79]
        >>> vol_2 = [4749, 6762, 4908, 2002, 9170, 9791]
        >>> d2 = pd.DataFrame(dict(price=close_2, volume=vol_2), index=idx)
        >>> d2
                                   price  volume
        2018-09-10 10:10:00+10:00  70.81    4749
        2018-09-10 10:11:00+10:00  70.78    6762
        2018-09-10 10:12:00+10:00  70.85    4908
        2018-09-10 10:13:00+10:00  70.79    2002
        2018-09-10 10:14:00+10:00  70.79    9170
        2018-09-10 10:15:00+10:00  70.79    9791
        >>> align_data(d1, d2)
                                   price_1  volume_1  price_2  volume_2
        2018-09-10 10:10:00+10:00    31.08     10166    70.81      4749
        2018-09-10 10:11:00+10:00    31.10     69981    70.78      6762
        2018-09-10 10:12:00+10:00    31.11     14343    70.85      4908
        2018-09-10 10:13:00+10:00    31.07     10096    70.79      2002
        2018-09-10 10:14:00+10:00    31.04     11506    70.79      9170
        2018-09-10 10:15:00+10:00    31.04      9718    70.79      9791
    """
    res = pd.DataFrame(pd.concat([
        d.loc[~d.index.duplicated(keep='first')].rename(
            columns=lambda vv: '%s_%d' % (vv, i + 1)
        ) for i, d in enumerate(args)
    ], axis=1))
    data_cols = [col for col in res.columns if col[-2:] == '_1']
    other_cols = [col for col in res.columns if col[-2:] != '_1']
    res.loc[:, other_cols] = res.loc[:, other_cols].fillna(method='pad')
    return res.dropna(subset=data_cols)