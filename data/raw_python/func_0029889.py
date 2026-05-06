def close_milestone(id, **kwargs):
    """
    Close a milestone. This triggers its release process.

    The user can optionally specify the release-date, otherwise today's date is
    used.

    If the wait parameter is specified and set to True, upon closing the milestone,
    we'll periodically check that the release being processed is done.

    Required:
    - id: int

    Optional:
    - wait key: bool
    """
    data = close_milestone_raw(id, **kwargs)
    if data:
        return utils.format_json(data)