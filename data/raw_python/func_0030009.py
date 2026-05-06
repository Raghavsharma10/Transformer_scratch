def reflect_filter(sources, model, cache=None):
    '''Returns the list of reflections of objects in the `source` list to other
    class. Objects that are not found in target table are silently discarded.
    '''
    targets = [reflect(source, model, cache=cache) for source in sources]
    # Some objects may not be available in target DB (not published), so we
    # have to exclude None from the list.
    return [target for target in targets if target is not None]