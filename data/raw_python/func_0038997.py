def dollarfy(x):
    """Replaces Math elements in element list 'x' with a $-enclosed string.

    stringify() passes through TeX math.  Use dollarfy(x) first to replace
    Math elements with math strings set in dollars.  'x' should be a deep copy
    so that the underlying document is left untouched.

    Returns 'x'."""

    def _dollarfy(key, value, fmt, meta):  # pylint: disable=unused-argument
        """Replaces Math elements"""
        if key == 'Math':
            return Str('$' + value[1] + '$')
        return None

    return walk(x, _dollarfy, '', {})