def pull_params(voevent):
    """
    Attempts to load the `What` section of a voevent as a nested dictionary.

    .. warning:: Deprecated due to `Missing name attributes` issues.

        `Param` or `Group` entries which are missing the `name` attribute
        will be entered under a dictionary key of ``None``. This means that if
        there are multiple entries missing the `name` attribute then earlier
        entries will be overwritten by later entries, so you will not be able
        to use this convenience routine effectively.
        Use :func:`get_grouped_params` and  :func:`get_toplevel_params`
        instead.

    Args:
        voevent (:class:`voeventparse.voevent.Voevent`): Root node of the VOevent etree.
    Returns:
        dict: Mapping of ``Group->Param->Attribs``.
        Access like so::

            foo_param_val = what_dict['GroupName']['ParamName']['value']

        .. note::

          Parameters without a group are indexed under the key 'None' - otherwise,
          we might get name-clashes between `params` and `groups` (unlikely but
          possible) so for ungrouped Params you'll need something like::

            what_dict[None]['ParamName']['value']

    """
    import warnings
    warnings.warn(
        """
        The function `pull_params` has been deprecated in favour of the split
        functions `get_toplevel_params` and `get_grouped_params`, due to 
        possible name-shadowing issues when combining multilevel-nested-dicts
        (see docs for details).
        
        This alias is preserved for backwards compatibility, and may be 
        removed in a future release.
        """,
        FutureWarning)
    result = OrderedDict()
    w = deepcopy(voevent.What)
    lxml.objectify.deannotate(w)
    if w.countchildren() == 0:
        return result
    toplevel_params = OrderedDict()
    result[None] = toplevel_params
    if w.find('Param') is not None:
        for p in w.Param:
            toplevel_params[p.attrib.get('name')] = p.attrib
    if w.find('Group') is not None:
        for g in w.Group:
            g_params = {}
            result[g.attrib.get('name')] = g_params
            if hasattr(g, 'Param'):
                for p in g.Param:
                    g_params[p.attrib.get('name')] = p.attrib
    return result