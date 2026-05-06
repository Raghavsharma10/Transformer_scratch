def get_toplevel_params(voevent):
    """
    Fetch ungrouped Params from the `What` section of a voevent as an omdict.

    This fetches 'toplevel' Params, i.e. those not enclosed in a Group element,
    and returns them as a nested dict-like structure, keyed like
    ParamName->AttribName.

    Note that since multiple Params may share the same ParamName, the returned
    data-structure is actually an
    `orderedmultidict.omdict <https://github.com/gruns/orderedmultidict>`_
    and has extra methods such as 'getlist' to allow retrieval of all values.

    Any Params with no defined name (technically off-spec, but not invalidated
    by the XML schema) are returned under the dict-key ``None``.

    Args:
        voevent (:class:`voeventparse.voevent.Voevent`): Root node of the VOevent etree.
    Returns (orderedmultidict.omdict):
        Mapping of ``ParamName->Attribs``.
        Typical access like so::

            foo_val = top_params['foo']['value']
            # If there are multiple Param entries named 'foo':
            all_foo_vals = [atts['value'] for atts in top_params.getlist('foo')]

    """
    result = OrderedDict()
    w = deepcopy(voevent.What)
    lxml.objectify.deannotate(w)
    return _get_param_children_as_omdict(w)