def gather_registries() -> Tuple[Dict, Mapping, Mapping]:
    """Get and clear the current |Node| and |Element| registries.

    Function |gather_registries| is thought to be used by class |Tester| only.
    """
    id2devices = copy.copy(_id2devices)
    registry = copy.copy(_registry)
    selection = copy.copy(_selection)
    dict_ = globals()
    dict_['_id2devices'] = {}
    dict_['_registry'] = {Node: {}, Element: {}}
    dict_['_selection'] = {Node: {}, Element: {}}
    return id2devices, registry, selection