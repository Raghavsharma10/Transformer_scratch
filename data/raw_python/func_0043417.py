def apply_preprocessors(root, src, dst, processors):
    """
    Preprocessors operate based on the source filename, and apply to each
    file individually.
    """
    matches = [(pattern, cmds) for pattern, cmds in processors.iteritems() if fnmatch(src, pattern)]
    if not matches:
        return False

    params = get_format_params(dst)

    for pattern, cmd_list in matches:
        for cmd in cmd_list:
            run_command(cmd, root=root, dst=dst, input=src, params=params)
            src = dst

    return True