def coerce_feature_branch(value):
    """
    Convert a string to a :class:`FeatureBranchSpec` object.

    :param value: A string or :class:`FeatureBranchSpec` object.
    :returns: A :class:`FeatureBranchSpec` object.
    """
    # Repository objects pass through untouched.
    if isinstance(value, FeatureBranchSpec):
        return value
    # We expect a string with a name or URL.
    if not isinstance(value, string_types):
        msg = "Expected string or FeatureBranchSpec object as argument, got %s instead!"
        raise ValueError(msg % type(value))
    return FeatureBranchSpec(expression=value)