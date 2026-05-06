def check(session, rev_id, page_id=None, radius=defaults.RADIUS,
          before=None, window=None, rvprop=None):
    """
    Checks the revert status of a revision.  With this method, you can
    determine whether an edit is a 'reverting' edit, was 'reverted' by another
    edit and/or was 'reverted_to' by another edit.

    :Parameters:
        session : :class:`mwapi.Session`
            An API session to make use of
        rev_id : int
            the ID of the revision to check
        page_id : int
            the ID of the page the revision occupies (slower if not provided)
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

    :Example:

        >>> import mwapi
        >>> import mwreverts.api
        >>>
        >>> session = mwapi.Session("https://en.wikipedia.org")
        >>>
        >>> def print_revert(revert):
        ...     if revert is None:
        ...         print(None)
        ...     else:
        ...         print(revert.reverting['revid'],
        ...               [r['revid'] for r in revert.reverteds],
        ...               revert.reverted_to['revid'])
        ...
        >>> reverting, reverted, reverted_to = \
        ...     mwreverts.api.check(session, 679778587)
        >>> print_revert(reverting)
        None
        >>> print_revert(reverted)
        679778743 [679778587] 679742862
        >>> print_revert(reverted_to)
        None

    """

    rev_id = int(rev_id)
    radius = int(radius)
    if radius < 1:
        raise TypeError("invalid radius.  Expected a positive integer.")

    page_id = int(page_id) if page_id is not None else None
    before = Timestamp(before) if before is not None else None

    rvprop = set(rvprop) if rvprop is not None else set()

    # If we don't have the page_id, we're going to need to look them up
    if page_id is None:
        page_id = get_page_id(session, rev_id)

    # Load history and current rev
    current_and_past_revs = list(n_edits_before(
        session,
        rev_id,
        page_id,
        n=radius + 1,
        rvprop={'ids', 'timestamp', 'sha1'} | rvprop
    ))

    if len(current_and_past_revs) < 1:
        raise KeyError("Revision {0} not found in page {1}."
                       .format(rev_id, page_id))

    current_rev, past_revs = (
        current_and_past_revs[-1],  # Current
        current_and_past_revs[:-1]  # Past revisions
    )

    if window is not None and before is None:
        before = Timestamp(current_rev['timestamp']) + window

    # Load future revisions
    future_revs = list(n_edits_after(
        session,
        rev_id + 1,
        page_id,
        n=radius,
        timestamp=before,
        rvprop={'ids', 'timestamp', 'sha1'} | rvprop
    ))

    return build_revert_tuple(
        rev_id, past_revs, current_rev, future_revs, radius)