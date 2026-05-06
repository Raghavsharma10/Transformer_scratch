def Group(params, name=None, type=None):
    """Groups together Params for adding under the 'What' section.

    Args:
        params(list of :func:`Param`): Parameter elements to go in this group.
        name(str): Group name. NB ``None`` is valid, since the group may be
            best identified by its type.
        type(str): Type of group, e.g. 'complex' (for real and imaginary).
    """
    atts = {}
    if name:
        atts['name'] = name
    if type:
        atts['type'] = type
    g = objectify.Element('Group', attrib=atts)
    for p in params:
        g.append(p)
    return g