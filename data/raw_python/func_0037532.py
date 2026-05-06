def get_grouped_params(voevent):
    """
    Fetch grouped Params from the `What` section of a voevent as an omdict.

    This fetches 'grouped' Params, i.e. those enclosed in a Group element,
    and returns them as a nested dict-like structure, keyed by
    GroupName->ParamName->AttribName.

    Note that since multiple Params may share the same ParamName, the returned
    data-structure is actually an
    `orderedmultidict.omdict <https://github.com/gruns/orderedmultidict>`_
    and has extra methods such as 'getlist' to allow retrieval of all values.

    Args:
        voevent (:class:`voeventparse.voevent.Voevent`): Root node of the VOevent etree.
    Returns (orderedmultidict.omdict):
        Mapping of ``ParamName->Attribs``.
        Typical access like so::

            foo_val = top_params['foo']['value']
            # If there are multiple Param entries named 'foo':
            all_foo_vals = [atts['value'] for atts in top_params.getlist('foo')]

    """
    groups_omd = OMDict()
    w = deepcopy(voevent.What)
    lxml.objectify.deannotate(w)
    if w.find('Group') is not None:
        for grp in w.Group:
            groups_omd.add(grp.attrib.get('name'),
                           _get_param_children_as_omdict(grp))
    return groups_omd