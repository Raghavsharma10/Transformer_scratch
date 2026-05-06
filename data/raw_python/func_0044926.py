def widthratio(value, maxvalue, max_width):
    """
    For creating bar charts and such, this tag calculates the ratio of a given
    value to a maximum value, and then applies that ratio to a constant.

    For example::

        <img src='bar.gif' height='10' width='{% widthratio this_value max_value 100 %}' />

    Above, if ``this_value`` is 175 and ``max_value`` is 200, the image in
    the above example will be 88 pixels wide (because 175/200 = .875;
    .875 * 100 = 87.5 which is rounded up to 88).
    """
    try:
        max_width = int(max_width)
    except ValueError:
        raise TemplateSyntaxError("widthratio final argument must be an number")
    try:
        value = float(value)
        maxvalue = float(maxvalue)
        ratio = (value / maxvalue) * max_width
    except (ValueError, ZeroDivisionError):
        return ''
    return str(int(round(ratio)))