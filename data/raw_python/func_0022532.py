def meld(*values):
    """Return the repeated value, or the first value if there's only one.

    This is a convenience function, equivalent to calling
    getvalue(repeated(x)) to get x.

    This function skips over instances of None in values (None is not allowed
    in repeated variables).

    Examples:
        meld("foo", "bar") # => ListRepetition("foo", "bar")
        meld("foo", "foo") # => ListRepetition("foo", "foo")
        meld("foo", None) # => "foo"
        meld(None) # => None
    """
    values = [x for x in values if x is not None]
    if not values:
        return None

    result = repeated(*values)
    if isrepeating(result):
        return result

    return getvalue(result)