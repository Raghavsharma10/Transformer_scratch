def begin_auth():
    """ Request authentication token to sign """

    repository    = request.headers['repository']
    if repository not in config['repositories']: return fail(no_such_repo_msg)

    # ==
    repository_path = config['repositories'][repository]['path']
    conn = auth_db_connect(cpjoin(repository_path, 'auth_transient.db')); gc_tokens(conn)

    # Issue a new token
    auth_token = base64.b64encode(pysodium.randombytes(35)).decode('utf-8')
    conn.execute("insert into tokens (expires, token, ip) values (?,?,?)",
                 (time.time() + 30, auth_token, request.environ['REMOTE_ADDR']))
    conn.commit()

    return success({'auth_token' : auth_token})