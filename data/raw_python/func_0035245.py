def step_impl10(context):
    """Create application list.

    :param context: test context.
    """
    assert context.app_list and len(
        context.app_list) > 0, "ENSURE: app list is provided."
    assert context.file_list and len(
        context.file_list) > 0, "ENSURE: file list is provided."
    context.fuzz_executor = FuzzExecutor(context.app_list, context.file_list)
    assert context.fuzz_executor, "VERIFY: fuzz executor created."