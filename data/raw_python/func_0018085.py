def handle_overrides(graph, overrides):
    """
    Handle any overrides for this model configuration.

    Parameters
    ----------
    graph : dict or object
        A dictionary (or an ObjectProxy) containing the object graph
        loaded from a YAML file.
    overrides : dict
        A dictionary containing overrides to apply. The location of
        the override is specified in the key as a dot-delimited path
        to the desired parameter, e.g. "model.corruptor.corruption_level".
    """
    for key in overrides:
        levels = key.split('.')
        part = graph
        for lvl in levels[:-1]:
            try:
                part = part[lvl]
            except KeyError:
                raise KeyError("'%s' override failed at '%s'", (key, lvl))
        try:
            part[levels[-1]] = overrides[key]
        except KeyError:
            raise KeyError("'%s' override failed at '%s'", (key, levels[-1]))