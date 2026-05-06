def prettify(amount, separator=','):
    """Separate with predefined separator."""
    orig = str(amount)
    new = re.sub("^(-?\d+)(\d{3})", "\g<1>{0}\g<2>".format(separator), str(amount))
    if orig == new:
        return new
    else:
        return prettify(new)