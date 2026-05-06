def update_handler(args):
    """usage: {program} update [<path>]

    Update out of date anchors in the current repository.
    """
    repo = _open_repo(args)

    for anchor_id, anchor in repo.items():
        try:
            anchor = update(anchor)
        except AlignmentError as e:
            print('Unable to update anchor {}. Reason: {}'.format(
                anchor_id, e))
        else:
            repo[anchor_id] = anchor