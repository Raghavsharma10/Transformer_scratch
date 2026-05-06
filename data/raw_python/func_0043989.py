def status_handler(args):
    """usage: {program} status [<path>]

    Validate the anchors in the current repository.
    """

    repo = _open_repo(args)

    for anchor_id, anchor in repo.items():
        diff_lines = get_anchor_diff(anchor)
        if diff_lines:
            print('{} {}:{} out-of-date'.format(
                anchor_id,
                anchor.file_path,
                anchor.context.offset))

    return ExitCode.OK