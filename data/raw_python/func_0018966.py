def reset_registries(dicts: Tuple[Dict, Mapping, Mapping]):
    """Reset the current |Node| and |Element| registries.

    Function |reset_registries| is thought to be used by class |Tester| only.
    """
    dict_ = globals()
    dict_['_id2devices'] = dicts[0]
    dict_['_registry'] = dicts[1]
    dict_['_selection'] = dicts[2]