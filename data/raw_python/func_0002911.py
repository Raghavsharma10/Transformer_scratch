def check_archive(schema, rev_id, namespace=None, title=None, timestamp=None,
                  radius=defaults.RADIUS,
                  before=None, window=None):
    """
    Checks the revert status of an archived revision (from a deleted page).
    With this method, you can determine whether an edit is a 'reverting'
    edit, was 'reverted' by another edit and/or was 'reverted_to' by
    another edit.

    :Parameters:
        session : :class:`mwapi.Session`
            An API session to make use of
        rev_id : int
            the ID of the revision to check
        namespace : int
            the namespace ID of the page the revision exists in
        title : str
            the title of the page the revision exists in
        timestamp : :class:`mwtypes.Timestamp`
            the timestamp that the revision for `rev_id` was saved
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
        A triple :class:`mwreverts.Revert`

        * reverting -- If this edit reverted other edit(s)
        * reverted -- If this edit was reverted by another edit
        * reverted_to -- If this edit was reverted to by another edit

    """

    rev_id = int(rev_id)
    radius = int(radius)
    if radius < 1:
        raise TypeError("invalid radius.  Expected a positive integer.")

    namespace = int(namespace) if namespace is not None else None
    title = str(title) if title is not None else None
    timestamp = Timestamp(timestamp) if timestamp is not None else None
    before = Timestamp(before) if before is not None else None

    # If we don't have the page_id, we're going to need to look them up
    if namespace is None or title is None or timestamp is None:
        namespace, title, timestamp = \
            get_archived_namespace_title_and_timestamp(schema, rev_id)

    # Load history and current rev
    current_and_past_revs = list(n_archived_edits_before(
        schema, rev_id + 1, namespace, title, timestamp + 1, n=radius + 1))

    if len(current_and_past_revs) < 1:
        raise KeyError("Revision {0} not found in page {1}(ns={2}) @ {3}."
                       .format(rev_id, title, namespace, timestamp))

    current_rev, past_revs = (
        current_and_past_revs[-1],  # Current rev is the last one returned
        current_and_past_revs[:-1]  # The rest are past revs
    )
    if current_rev.ar_rev_id != rev_id:
        raise KeyError("Revision {0} not found in page {1}(ns={2}) @ {3}."
                       .format(rev_id, title, namespace, timestamp))

    if window is not None and before is None:
        before = Timestamp(current_rev.ar_timestamp) + window

    # Load future revisions
    future_revs = list(n_archived_edits_after(
        schema, rev_id, namespace, title, timestamp, n=radius, before=before))

    return build_revert_tuple(
        rev_id, past_revs, current_rev, future_revs, radius)