def ts_bin_op(op_name, ts1, ts2, all=True, fill=None, name=None):
    '''Entry point for any arithmetic type function performed on a timeseries
    and/or a scalar.
    op_name - name of the function to be performed
    ts1, ts2 - timeseries or scalars that the function is to performed over
    all - whether all dates should be included in the result
    fill - the value that should be used to represent "missing values"
    name - the name of the resulting time series
    '''
    op = op_get(op_name)
    fill = fill if fill is not None else settings.missing_value
    if hasattr(fill, '__call__'):
        fill_fn = fill
    else:
        fill_fn = lambda: fill

    name = name or '%s(%s,%s)' % (op_name, ts1, ts2)
    if is_timeseries(ts1):
        ts = ts1
        if is_timeseries(ts2):
            dts, data = op_ts_ts(op_name, op, ts1, ts2, all, fill_fn)

        else:
            dts, data = op_ts_scalar(op_name, op, ts1, ts2, fill_fn)
    else:
        if is_timeseries(ts2):
            ts = ts2
            dts, data = op_scalar_ts(op_name, op, ts1, ts2, fill_fn)
        else:
            return op(ts1, ts2)

    return ts.clone(date=dts, data=data, name=name)