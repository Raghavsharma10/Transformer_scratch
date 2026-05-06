def _get_anchor(repo, id_prefix):
    """Get an anchor by ID, or a prefix of its id.
    """
    result = None
    for anchor_id, anchor in repo.items():
        if anchor_id.startswith(id_prefix):
            if result is not None:
                raise ExitError(
                    ExitCode.DATA_ERR,
                    'Ambiguous ID specification')

            result = (anchor_id, anchor)

    if result is None:
        raise ExitError(
            ExitCode.DATA_ERR,
            'No anchor matching ID specification')

    return result