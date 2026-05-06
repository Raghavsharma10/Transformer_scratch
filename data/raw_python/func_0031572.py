def format_float(digit=0, is_pct=False):
    """
    Number display format for pandas

    Args:
        digit: number of digits to keep
               if negative, add one space in front of positive pct
        is_pct: % display

    Returns:
        lambda function to format floats

    Examples:
        >>> format_float(0)(1e5)
        '100,000'
        >>> format_float(1)(1e5)
        '100,000.0'
        >>> format_float(-1, True)(.2)
        ' 20.0%'
        >>> format_float(-1, True)(-.2)
        '-20.0%'
        >>> pd.options.display.float_format = format_float(2)
    """
    if is_pct:
        space = ' ' if digit < 0 else ''
        fmt = f'{{:{space}.{abs(int(digit))}%}}'
        return lambda vv: 'NaN' if np.isnan(vv) else fmt.format(vv)

    else:
        return lambda vv: 'NaN' if np.isnan(vv) else (
            f'{{:,.{digit}f}}'.format(vv) if vv else '-' + ' ' * abs(digit)
        )