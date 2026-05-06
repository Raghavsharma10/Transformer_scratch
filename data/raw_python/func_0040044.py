def push_file():
    """ Push a file to the server """ #NOTE beware that reading post data in flask causes hang until file upload is complete

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

        # There is no valid reason for path traversal characters to be in a file path within this system
        file_path = request.headers['path']
        if any(True for item in re.split(r'\\|/', file_path) if item in ['..', '.']): return fail()

        #===
        tmp_path = cpjoin(repository_path, 'tmp_file')
        with open(tmp_path, 'wb') as f:
            while True:
                chunk = request.stream.read(1000 * 1000)
                if chunk == b'': break
                f.write(chunk)

        #===
        data_store.fs_put_from_file(tmp_path, {'path' : file_path})

        # updates the user lock expiry
        update_user_lock(repository_path, session_token)
        return success()

    return lock_access(repository_path, with_exclusive_lock)