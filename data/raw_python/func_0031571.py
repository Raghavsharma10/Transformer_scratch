def spline_curve(x, y, step, val_min=0, val_max=None, kind='quadratic', **kwargs):
    """
    Fit spline curve for given x, y values

    Args:
        x: x-values
        y: y-values
        step: step size for interpolation
        val_min: minimum value of result
        val_max: maximum value of result
        kind: for scipy.interpolate.interp1d
        Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’,
        ‘quadratic’, ‘cubic’, ‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’
        refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and
        ‘next’ simply return the previous or next value of the point) or as an integer specifying
        the order of the spline interpolator to use. Default is ‘linear’.
        **kwargs: additional parameters for interp1d

    Returns:
        pd.Series: fitted curve

    Examples:
        >>> x = pd.Series([1, 2, 3])
        >>> y = pd.Series([np.exp(1), np.exp(2), np.exp(3)])
        >>> r = spline_curve(x=x, y=y, step=.5, val_min=3, val_max=18, fill_value='extrapolate')
        >>> r.round(2).index.tolist()
        [1.0, 1.5, 2.0, 2.5, 3.0]
        >>> r.round(2).tolist()
        [3.0, 4.05, 7.39, 12.73, 18.0]
        >>> y_df = pd.DataFrame(dict(a=[np.exp(1), np.exp(2), np.exp(3)], b=[2, 3, 4]))
        >>> r_df = spline_curve(x=x, y=y_df, step=.5, val_min=3, fill_value='extrapolate')
        >>> r_df.round(2)
                 a    b
        1.00  3.00 3.00
        1.50  4.05 3.00
        2.00  7.39 3.00
        2.50 12.73 3.50
        3.00 20.09 4.00
    """
    from scipy.interpolate import interp1d
    from collections import OrderedDict

    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(OrderedDict([(col, spline_curve(
            x, y.loc[:, col], step=step, val_min=val_min, val_max=val_max, kind=kind
        )) for col in y.columns]))
    fitted_curve = interp1d(x, y, kind=kind, **kwargs)
    new_x = np.arange(x.min(), x.max() + step / 2., step=step)
    return pd.Series(
        new_x, index=new_x, name=y.name if hasattr(y, 'name') else None
    ).apply(fitted_curve).clip(val_min, val_max)