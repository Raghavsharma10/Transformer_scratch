def step_impl08(context):
    """Create file list.

    :param context: test context.
    """
    assert context.table, "ENSURE: table is provided."
    context.file_list = [row['file_path'] for row in context.table.rows]