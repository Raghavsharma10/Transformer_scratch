def delete_files():
    """ Delete one or more files from the server """

    session_token = request.headers['session_token']
    repository    = request.headers['repository']

    #===
    current_user = have_authenticated_user(request.environ['REMOTE_ADDR'], repository, session_token)
    if current_user is False: return fail(user_auth_fail_msg)

    #===
    repository_path = config['repositories'][repository]['path']
    body_data = request.get_json()

    def with_exclusive_lock():
        if not varify_user_lock(repository_path, session_token): return fail(lock_fail_msg)

        try:
            data_store = versioned_storage(repository_path)
            if not data_store.have_active_commit(): return fail(no_active_commit_msg)

            #-------------
            for fle in json.loads(body_data['files']):
                data_store.fs_delete(fle)

            # updates the user lock expiry
            update_user_lock(repository_path, session_token)
            return success()
        except Exception: return fail() # pylint: disable=broad-except
    return lock_access(repository_path, with_exclusive_lock)