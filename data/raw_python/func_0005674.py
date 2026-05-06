def progressed_bar(count, total=100, status=None, suffix=None, bar_len=10):
    """render a progressed.io like progress bar"""
    status = status or ''
    suffix = suffix or '%'
    assert isinstance(count, int)
    count_normalized = count if count <= total else total
    filled_len = int(round(bar_len * count_normalized / float(total)))
    percents = 100.0 * count / float(total)
    color = '#5cb85c'
    if percents < 30.0:
        color = '#d9534f'
    if percents < 70.0:
        color = '#f0ad4e'
    text_color = colors.fg(color)
    bar_color = text_color + colors.bg(color)
    nc_color = colors.dark_gray
    progressbar = (colors.bg('#428bca') | status) if status else ''
    progressbar += (bar_color | ('█' * filled_len))
    progressbar += (nc_color | ('█' * (bar_len - filled_len)))
    progressbar += (text_color | (str(count) + suffix))
    return progressbar