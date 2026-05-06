def authenticate():
    """ This does two things, either validate a pre-existing session token
    or create a new one from a signed authentication token. """

    client_ip     = request.environ['REMOTE_ADDR']
    repository    = request.headers['repository']
    if repository not in config['repositories']: return fail(no_such_repo_msg)

    # ==
    repository_path = config['repositories'][repository]['path']
    conn = auth_db_connect(cpjoin(repository_path, 'auth_transient.db')); gc_tokens(conn)
    gc_tokens(conn)

    # Allow resume of an existing session
    if 'session_token' in request.headers:
        session_token = request.headers['session_token']

        conn.execute("delete from session_tokens where expires < ?", (time.time(),)); conn.commit()
        res = conn.execute("select * from session_tokens where token = ? and ip = ?", (session_token, client_ip)).fetchall()
        if res != []: return success({'session_token'  : session_token})
        else:         return fail(user_auth_fail_msg)

    # Create a new session
    else:
        user       = request.headers['user']
        auth_token = request.headers['auth_token']
        signiture  = request.headers['signature']

        try:
            public_key = config['users'][user]['public_key']

            # signature
            pysodium.crypto_sign_verify_detached(base64.b64decode(signiture), auth_token, base64.b64decode(public_key))

            # check token was previously issued by this system and is still valid
            res = conn.execute("select * from tokens where token = ? and ip = ? ", (auth_token, client_ip)).fetchall()

            # Validate token matches one we sent
            if res == [] or len(res) > 1: return fail(user_auth_fail_msg)

            # Does the user have permission to use this repository?
            if repository not in config['users'][user]['uses_repositories']: return fail(user_auth_fail_msg)

            # Everything OK
            conn.execute("delete from tokens where token = ?", (auth_token,)); conn.commit()

            # generate a session token and send it to the client
            session_token = base64.b64encode(pysodium.randombytes(35))
            conn.execute("insert into session_tokens (token, expires, ip, username) values (?,?,?, ?)",
                         (session_token, time.time() + extend_session_duration, client_ip, user))
            conn.commit()
            return success({'session_token'  : session_token})

        except Exception: # pylint: disable=broad-except
            return fail(user_auth_fail_msg)