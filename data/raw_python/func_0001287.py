def write_authorized_keys(user=None):
    """Write public keys back to authorized_keys file. Create keys directory if it doesn't already exist.

    args:
        user (User): Instance of User containing keys.

    returns:
        list: Authorised keys for the specified user.
    """
    authorized_keys = list()
    authorized_keys_dir = '{0}/.ssh'.format(os.path.expanduser('~{0}'.format(user.name)))
    rnd_chars = random_string(length=RANDOM_FILE_EXT_LENGTH)
    authorized_keys_path = '{0}/authorized_keys'.format(authorized_keys_dir)
    tmp_authorized_keys_path = '/tmp/authorized_keys_{0}_{1}'.format(user.name, rnd_chars)

    if not os.path.isdir(authorized_keys_dir):
        execute_command(shlex.split(str('{0} mkdir -p {1}'.format(sudo_check(), authorized_keys_dir))))
    for key in user.public_keys:
        authorized_keys.append('{0}\n'.format(key.raw))
    with open(tmp_authorized_keys_path, mode=text_type('w+')) as keys_file:
        keys_file.writelines(authorized_keys)
    execute_command(
        shlex.split(str('{0} cp {1} {2}'.format(sudo_check(), tmp_authorized_keys_path, authorized_keys_path))))
    execute_command(shlex.split(str('{0} chown -R {1} {2}'.format(sudo_check(), user.name, authorized_keys_dir))))
    execute_command(shlex.split(str('{0} chmod 700 {1}'.format(sudo_check(), authorized_keys_dir))))
    execute_command(shlex.split(str('{0} chmod 600 {1}'.format(sudo_check(), authorized_keys_path))))
    execute_command(shlex.split(str('{0} rm {1}'.format(sudo_check(), tmp_authorized_keys_path))))