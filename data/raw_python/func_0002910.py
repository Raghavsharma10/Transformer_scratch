def check(schema, rev_id, page_id=None, radius=defaults.RADIUS,
          before=None, window=None):
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

        >>> import mwdb
        >>> import mwreverts.api
        >>>
        >>> schema = mwdb.Schema("mysql+pymysql://enwiki.labsdb/enwiki_p" +
                                 "?read_default_file=~/replica.my.cnf")
        >>>
        >>> def print_revert(revert):
        ...     if revert is None:
        ...         print(None)
        ...     else:
        ...         print(revert.reverting['rev_id'],
        ...               [r['rev_id'] for r in revert.reverteds],
        ...               revert.reverted_to['rev_id'])
        ...
        >>> reverting, reverted, reverted_to = \\
        ...     mwreverts.db.check(schema, 679778587)
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

    # If we don't have the page_id, we're going to need to look them up
    if page_id is None:
        page_id = get_page_id(schema, rev_id)

    # Load history and current rev
    current_and_past_revs = list(n_edits_before(
        schema, rev_id + 1, page_id, n=radius + 1))

    if len(current_and_past_revs) < 1:
        raise KeyError("Revision {0} not found in page {1}."
                       .format(rev_id, page_id))

    current_rev, past_revs = (
        current_and_past_revs[-1],  # Current rev is the last one returned
        current_and_past_revs[:-1]  # The rest are past revs
    )
    if current_rev.rev_id != rev_id:
        raise KeyError("Revision {0} not found in page {1}."
                       .format(rev_id, page_id))

    if window is not None and before is None:
        before = Timestamp(current_rev.rev_timestamp) + window

    # Load future revisions
    future_revs = list(n_edits_after(
        schema, rev_id, page_id, n=radius, before=before))

    return build_revert_tuple(
        rev_id, past_revs, current_rev, future_revs, radius)