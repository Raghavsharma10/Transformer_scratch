def elt(eltType, numargs):  # pylint: disable=invalid-name
    """Returns Element(*value) to create pandoc json elements.

    This should be used in place of pandocfilters.elt().  This version
    ensures that the content is stored in a list, not a tuple.
    """
    def Element(*value):  # pylint: disable=invalid-name
        """Creates an element."""
        el = _elt(eltType, numargs)(*value)
        if isinstance(el['c'], tuple):
            el['c'] = list(el['c'])  # The content should be a list, not tuple
        return el
    return Element