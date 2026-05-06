def update_user_lock(repository_path, session_token):
    """ Write or clear the user lock file """ # NOTE ALWAYS use within lock access callback

    # While the user lock file should ALWAYS be written only within a lock_access
    # callback, it is sometimes read asynchronously. Because of this updates to
    # the file must be atomic. Write plus move is used to achieve this.
    real_path = cpjoin(repository_path, 'user_file')
    tmp_path  = cpjoin(repository_path, 'new_user_file')

    with open(tmp_path, 'w') as fd2:
        if session_token is None: fd2.write('')
        else: fd2.write(json.dumps({'session_token' : session_token, 'expires' : int(time.time()) + 30}))
        fd2.flush()
    os.rename(tmp_path, real_path)