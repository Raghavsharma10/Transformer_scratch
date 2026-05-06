def detach_attrs_factory(f):
    """Returns detach_attrs(key, value, fmt, meta) action that detaches
    attributes attached to elements of type f (e.g. pandocfilters.Math, etc).
    Attributes provided natively by pandoc will be left as is."""

    # Get the name and standard length
    name = f.__closure__[0].cell_contents
    n = f.__closure__[1].cell_contents

    def detach_attrs(key, value, fmt, meta):  # pylint: disable=unused-argument
        """Detaches the attributes."""
        if key == name:
            assert len(value) <= n+1
            if len(value) == n+1:
                # Make sure value[0] represents attributes then delete
                assert len(value[0]) == 3
                assert isinstance(value[0][0], STRTYPES)
                assert isinstance(value[0][1], list)
                assert isinstance(value[0][2], list)
                del value[0]

    return detach_attrs