def commit():
    """ Commit changes and release the write lock """

    session_token = request.headers['session_token']
    repository    = request.headers['repository']

    #===
    current_user = have_authenticated_user(request.environ['REMOTE_ADDR'], repository, session_token)
    if current_user is False: return fail(user_auth_fail_msg)

    #===
    repository_path = config['repositories'][repository]['path']

    def with_exclusive_lock():
        if not varify_user_lock(repository_path, session_token): return fail(lock_fail_msg)

        #===
        data_store = versioned_storage(repository_path)
        if not data_store.have_active_commit(): return fail(no_active_commit_msg)

        result = {}
        if request.headers['mode'] == 'commit':
            new_head = data_store.commit(request.headers['commit_message'], current_user['username'])
            result = {'head' : new_head}
        else:
            data_store.rollback()

        # Release the user lock
        update_user_lock(repository_path, None)
        return success(result)
    return lock_access(repository_path, with_exclusive_lock)