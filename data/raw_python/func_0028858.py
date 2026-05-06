def fake_lens_path_view(lens_path, obj):
    """
    Simulates R.view with a lens_path since we don't have lens functions
    :param lens_path: Array of string paths
    :param obj: Object containing the given path
    :return: The value at the path or None
    """
    segment = head(lens_path)
    return if_else(
        both(lambda _: identity(segment), has(segment)),
        # Recurse on the rest of the path
        compose(fake_lens_path_view(tail(lens_path)), getitem(segment)),
        # Give up
        lambda _: None
    )(obj)