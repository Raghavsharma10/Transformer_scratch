def have_authenticated_user(client_ip, repository, session_token):
    """ check user submitted session token against the db and that ip has not changed """

    if repository not in config['repositories']: return False

    repository_path = config['repositories'][repository]['path']
    conn = auth_db_connect(cpjoin(repository_path, 'auth_transient.db'))

    # Garbage collect session tokens. We must not garbage collect the authentication token of the client
    # which is currently doing a commit. Large files can take a long time to upload and during this time,
    # the locks expiration is not being updated thus can expire. This is a problem here as session tokens
    # table is garbage collected every time a user authenticates. It does not matter if the user_lock
    # expires while the client also holds the flock, as it is updated to be in the future at the end of
    # the current operation. We exclude any tokens owned by the client which currently owns the user
    # lock for this reason.
    user_lock = read_user_lock(repository_path)
    active_commit = user_lock['session_token'] if user_lock != None else None

    if active_commit != None: conn.execute("delete from session_tokens where expires < ? and token != ?", (time.time(), active_commit))
    else:                     conn.execute("delete from session_tokens where expires < ?", (time.time(),))

    # Get the session token
    res = conn.execute("select * from session_tokens where token = ? and ip = ?", (session_token, client_ip)).fetchall()

    if res != [] and repository in config['users'][res[0]['username']]['uses_repositories']:
        conn.execute("update session_tokens set expires = ? where token = ? and ip = ?",
                     (time.time() + extend_session_duration, session_token, client_ip))

        conn.commit() # to make sure the update and delete have the same view

        return res[0]

    conn.commit()
    return False