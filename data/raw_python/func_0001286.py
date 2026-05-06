def read_authorized_keys(username=None):
    """Read public keys from specified user's authorized_keys file.

    args:
        username (str): username.

    returns:
        list: Authorised keys for the specified user.
    """
    authorized_keys_path = '{0}/.ssh/authorized_keys'.format(os.path.expanduser('~{0}'.format(username)))
    rnd_chars = random_string(length=RANDOM_FILE_EXT_LENGTH)
    tmp_authorized_keys_path = '/tmp/authorized_keys_{0}_{1}'.format(username, rnd_chars)
    authorized_keys = list()
    copy_result = execute_command(
        shlex.split(str('{0} cp {1} {2}'.format(sudo_check(), authorized_keys_path, tmp_authorized_keys_path))))
    result_message = copy_result[0][1].decode('UTF-8')
    if 'you must have a tty to run sudo' in result_message:  # pragma: no cover
        raise OSError("/etc/sudoers is blocked sudo. Remove entry: 'Defaults    requiretty'.")
    elif 'No such file or directory' not in result_message:
        execute_command(shlex.split(str('{0} chmod 755 {1}'.format(sudo_check(), tmp_authorized_keys_path))))
        with open(tmp_authorized_keys_path) as keys_file:
            for key in keys_file:
                authorized_keys.append(PublicKey(raw=key))
        execute_command(shlex.split(str('{0} rm {1}'.format(sudo_check(), tmp_authorized_keys_path))))
    return authorized_keys