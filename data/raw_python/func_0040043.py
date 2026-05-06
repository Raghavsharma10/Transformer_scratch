def begin_commit():
    """ Allow a client to begin a commit and acquire the write lock """

    session_token = request.headers['session_token']
    repository    = request.headers['repository']

    #===
    current_user = have_authenticated_user(request.environ['REMOTE_ADDR'], repository, session_token)
    if current_user is False: return fail(user_auth_fail_msg)

    #===
    repository_path = config['repositories'][repository]['path']

    def with_exclusive_lock():
        # The commit is locked for a given time period to a given session token,
        # a client must hold this lock to use any of push_file(), delete_files() and commit().
        # It does not matter if the user lock technically expires while a client is writing
        # a large file, as the user lock is locked using flock for the duration of any
        # operation and thus cannot be stolen by another client. It is updated to be in
        # the future before returning to the client. The lock only needs to survive until
        # the client owning the lock sends another request and re acquires the flock.
        if not can_aquire_user_lock(repository_path, session_token): return fail(lock_fail_msg)

        # Commits can only take place if the committing user has the latest revision,
        # as committing from an outdated state could cause unexpected results, and may
        # have conflicts. Conflicts are resolved during a client update so they are
        # handled by the client, and a server interface for this is not needed.
        data_store = versioned_storage(repository_path)
        if data_store.get_head() != request.headers["previous_revision"]: return fail(need_to_update_msg)


        # Should the lock expire, the client which had the lock previously will be unable
        # to continue the commit it had in progress. When this, or another client attempts
        # to commit again it must do so by first obtaining the lock again by calling begin_commit().
        # Any remaining commit data from failed prior commits is garbage collected here.
        # While it would technically be possible to implement commit resume should the same
        # client resume, I only see commits failing due to a network error and this is so
        # rare I don't think it's worth the trouble.
        if data_store.have_active_commit(): data_store.rollback()

        #------------
        data_store.begin()
        update_user_lock(repository_path, session_token)

        return success()
    return lock_access(repository_path, with_exclusive_lock)