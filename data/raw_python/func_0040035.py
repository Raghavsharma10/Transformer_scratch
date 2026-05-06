def can_aquire_user_lock(repository_path, session_token):
    """ Allow a user to acquire the lock if no other user is currently using it, if the original
    user is returning, presumably after a network error, or if the lock has expired.  """
    # NOTE ALWAYS use within lock access callback

    user_file_path = cpjoin(repository_path, 'user_file')
    if not os.path.isfile(user_file_path): return True
    with open(user_file_path, 'r') as fd2:
        content = fd2.read()
        if len(content) == 0: return True
        try: res = json.loads(content)
        except ValueError: return True
        if res['expires'] < int(time.time()): return True
        elif res['session_token'] == session_token: return True
    return False