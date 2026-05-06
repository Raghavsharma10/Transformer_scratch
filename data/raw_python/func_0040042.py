def pull_file():
    """ Get a file from the server """

    session_token = request.headers['session_token']
    repository    = request.headers['repository']

    #===
    current_user = have_authenticated_user(request.environ['REMOTE_ADDR'], repository, session_token)
    if current_user is False: return fail(user_auth_fail_msg)


    #===
    data_store = versioned_storage(config['repositories'][repository]['path'])
    file_info = data_store.get_file_info_from_path(request.headers['path'])

    return success({'file_info_json' : json.dumps(file_info)},
                   send_from_directory(data_store.get_file_directory_path(file_info['hash']), file_info['hash'][2:]))