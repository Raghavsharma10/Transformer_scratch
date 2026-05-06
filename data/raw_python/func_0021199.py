def build_index_name(app, *parts):
    """Build an index name from parts.

    :param parts: Parts that should be combined to make an index name.
    """
    base_index = os.path.splitext(
        '-'.join([part for part in parts if part])
    )[0]

    return prefix_index(app=app, index=base_index)