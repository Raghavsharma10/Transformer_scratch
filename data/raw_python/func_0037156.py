def EventIvorn(ivorn, cite_type):
    """
    Used to cite earlier VOEvents.

    Use in conjunction with :func:`.add_citations`

    Args:
        ivorn(str): It is assumed this will be copied verbatim from elsewhere,
            and so these should have any prefix (e.g. 'ivo://','http://')
            already in place - the function will not alter the value.
        cite_type (:class:`.definitions.cite_types`): String conforming to one
            of the standard citation types.

    """
    # This is an ugly hack around the limitations of the  lxml.objectify API:
    c = objectify.StringElement(cite=cite_type)
    c._setText(ivorn)
    c.tag = "EventIVORN"
    return c