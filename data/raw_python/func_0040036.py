def varify_user_lock(repository_path, session_token):
    """ Verify that a returning user has a valid token and their lock has not expired """

    with open(cpjoin(repository_path, 'user_file'), 'r') as fd2:
        content = fd2.read()
        if len(content) == 0: return False
        try: res = json.loads(content)
        except ValueError: return False
        return res['session_token'] == session_token and int(time.time()) < int(res['expires'])
    return False