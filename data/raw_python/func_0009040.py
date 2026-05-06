def keys_by_creation(group):
    """Returns a sequence of links in group in order of creation.

    Raises an error if the group was not set to track creation order.

    """
    from h5py import h5
    out = []
    try:
        group._id.links.iterate(
            out.append, idx_type=h5.INDEX_CRT_ORDER, order=h5.ITER_INC)
    except (AttributeError, RuntimeError):
        # pre 2.2 shim
        def f(name):
            if name.find(b'/', 1) == -1:
                out.append(name)
        group._id.links.visit(
            f, idx_type=h5.INDEX_CRT_ORDER, order=h5.ITER_INC)
    return map(group._d, out)