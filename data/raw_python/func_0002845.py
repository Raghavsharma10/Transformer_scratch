def check_deleted(session, rev_id, title=None, timestamp=None,
                  radius=defaults.RADIUS, before=None, window=None,
                  rvprop=None):
    """
    Checks the revert status of a deleted revision.  With this method, you can
    determine whether an edit is a 'reverting' edit, was 'reverted' by another
    edit and/or was 'reverted_to' by another edit.

    :Parameters:
        session : :class:`mwapi.Session`
            An API session to make use of
        rev_id : int
            the ID of the revision to check
        title : str
            the title of the page the revision occupies (slower if not
            provided) Note that the MediaWiki API expects the title to
            include the namespace prefix (e.g. "User_talk:EpochFail")
        radius : int
            a positive integer indicating the maximum number of revisions
            that can be reverted
        before : :class:`mwtypes.Timestamp`
            if set, limits the search for *reverting* revisions to those which
            were saved before this timestamp
        window : int
            if set, limits the search for *reverting* revisions to those which
            were saved within `window` seconds after the reverted edit
        rvprop : set( str )
            a set of properties to include in revisions

    :Returns:
        A triple :class:`mwreverts.Revert` | `None`

        * reverting -- If this edit reverted other edit(s)
        * reverted -- If this edit was reverted by another edit
        * reverted_to -- If this edit was reverted to by another edit
    """

    rev_id = int(rev_id)
    radius = int(radius)
    if radius < 1:
        raise TypeError("invalid radius.  Expected a positive integer.")

    title = str(title) if title is not None else None
    before = Timestamp(before) if before is not None else None

    rvprop = set(rvprop) if rvprop is not None else set()

    # If we don't have the title, we're going to need to look it up
    if title is None or timestamp is None:
        title, timestamp = get_deleted_title_and_timestamp(session, rev_id)

    # Load history and current rev
    current_and_past_revs = list(n_deleted_edits_before(
        session, rev_id, title, timestamp, n=radius + 1,
        rvprop={'ids', 'timestamp', 'sha1'} | rvprop
    ))

    if len(current_and_past_revs) < 1:
        raise KeyError("Revision {0} not found in page {1}."
                       .format(rev_id, title))

    current_rev, past_revs = (
        current_and_past_revs[-1],  # Current
        current_and_past_revs[:-1]  # Past revisions
    )

    if window is not None and before is None:
        before = Timestamp(current_rev['timestamp']) + window

    # Load future revisions
    future_revs = list(n_deleted_edits_after(
        session, rev_id + 1, title, timestamp, n=radius, before=before,
        rvprop={'ids', 'timestamp', 'sha1'} | rvprop
    ))

    return build_revert_tuple(
        rev_id, past_revs, current_rev, future_revs, radius)