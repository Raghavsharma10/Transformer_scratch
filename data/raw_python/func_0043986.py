def remove_handler(args):
    """usage: {program} remove <anchor-id> [<path>]

    Remove an existing anchor.
    """

    repo = _open_repo(args)
    anchor_id, anchor = _get_anchor(repo, args['<anchor-id>'])
    del repo[anchor_id]

    return ExitCode.OK