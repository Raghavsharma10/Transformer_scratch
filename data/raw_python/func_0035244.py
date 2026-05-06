def step_impl09(context):
    """Create application list.

    :param context: test context.
    """
    assert context.table, "ENSURE: table is provided."
    context.app_list = [row['application'] for row in context.table.rows]