def _unblast(name2vals, name_map):
    """Helper function to lift str -> bool maps used by aiger
    to the word level. Dual of the `_blast` function."""
    def _collect(names):
        return tuple(name2vals[n] for n in names)

    return {bvname: _collect(names) for bvname, names in name_map}