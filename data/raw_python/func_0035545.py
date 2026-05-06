def step_impl06(context):
    """Prepare test for singleton property.

    :param context: test context.
    """
    store = context.SingleStore
    context.st_1 = store()
    context.st_2 = store()
    context.st_3 = store()