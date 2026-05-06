def quotify(x):
    """Replaces Quoted elements in element list 'x' with quoted strings.

    Pandoc uses the Quoted element in its json when --smart is enabled.
    Output to TeX/pdf automatically triggers --smart.

    stringify() ignores Quoted elements.  Use quotify() first to replace
    Quoted elements in 'x' with quoted strings.  'x' should be a deep copy so
    that the underlying document is left untouched.

    Returns x."""

    def _quotify(key, value, fmt, meta):  # pylint: disable=unused-argument
        """Replaced Quoted elements with quoted strings."""
        if key == 'Quoted':
            ret = []
            quote = '"' if value[0]['t'] == 'DoubleQuote' else "'"
            if value[1][0]['t'] == 'Str':
                value[1][0]['c'] = quote + value[1][0]['c']
            else:
                ret.append(Str(quote))

            if value[1][-1]['t'] == 'Str':
                value[1][-1]['c'] = value[1][-1]['c'] + quote
                ret += value[1]
            else:
                ret += value[1] + [Str(quote)]
            return ret
        return None

    return walk(walk(x, _quotify, '', {}), join_strings, '', {})