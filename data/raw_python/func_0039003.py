def repair_refs(key, value, fmt, meta):  # pylint: disable=unused-argument
    """Using "-f markdown+autolink_bare_uris" with pandoc < 1.18 splits a
    reference like "{@fig:one}" into email Link and Str elements.  This
    function replaces the mess with the Cite and Str elements we normally
    get.  Call this before any reference processing."""

    if _PANDOCVERSION >= '1.18':
        return

    # The problem spans multiple elements, and so can only be identified in
    # element lists.  Element lists are encapsulated in different ways.  We
    # must process them all.

    if key in ('Para', 'Plain'):
        _repair_refs(value)
    elif key == 'Image':
        _repair_refs(value[-2])
    elif key == 'Table':
        _repair_refs(value[-5])