def step_impl07(context):
    """Test for singleton property.

    :param context: test context.
    """
    assert context.st_1 is context.st_2
    assert context.st_2 is context.st_3