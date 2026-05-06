def _set_value(instance_to_path_map, path_to_instance_map, prop_tree, config_instance):
    """ Finds appropriate term in the prop_tree and sets its value from config_instance.

    Args:
        configs_map (dict): key is id of the config, value is Config instance (AKA cache of the configs)
        prop_tree (PropertyDictTree): poperty tree to populate.
        config_instance (Config):

    """
    path = instance_to_path_map[config_instance]

    # find group
    group = prop_tree
    for elem in path[:-1]:
        group = getattr(group, elem)

    assert group._key == config_instance.parent.key
    setattr(group, config_instance.key, config_instance.value)

    #
    # bind config to the term
    #
    # FIXME: Make all the terms to store config instance the same way.
    term = getattr(group, config_instance.key)
    try:
        if hasattr(term, '_term'):
            # ScalarTermS and ScalarTermU case
            term._term._config = config_instance
            return
    except KeyError:
        # python3 case. TODO: Find the way to make it simple.
        pass

    try:
        if hasattr(term, '_config'):
            term._config = config_instance
            return
    except KeyError:
        # python3 case. TODO: Find the way to make it simple.
        pass
    else:
        pass