def diff_handler(args):
    """usage: {program} diff <anchor-id>

    Show the difference between an anchor and the current state of the source.
    """

    repo = _open_repo(args)
    anchor_id, anchor = _get_anchor(repo, args['<anchor-id>'])

    diff_lines = get_anchor_diff(anchor)
    sys.stdout.writelines(diff_lines)

    return ExitCode.OK