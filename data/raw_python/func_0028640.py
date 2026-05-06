def get_properties(zos_obj):
    """Returns a lists of properties bound to the object `zos_obj`

    @param zos_obj: ZOS API Python COM object
    @return prop_get: list of properties that are only getters
    @return prop_set: list of properties that are both getters and setters
    """
    prop_get = set(zos_obj._prop_map_get_.keys())
    prop_set = set(zos_obj._prop_map_put_.keys())
    if prop_set.issubset(prop_get):
        prop_get = prop_get.difference(prop_set)
    else:
        msg = 'Assumption all getters are also setters is incorrect!'
        raise NotImplementedError(msg)
    return list(prop_get), list(prop_set)