def _blast(bvname2vals, name_map):
    """Helper function to expand (blast) str -> int map into str ->
    bool map. This is used to send word level inputs to aiger."""
    if len(name_map) == 0:
        return dict()
    return fn.merge(*(dict(zip(names, bvname2vals[bvname]))
                      for bvname, names in name_map))