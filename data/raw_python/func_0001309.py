def read_sudoers():
    """ Read the sudoers entry for the specified user.

    args:
        username (str): username.

    returns:`r
        str: sudoers entry for the specified user.
    """
    sudoers_path = '/etc/sudoers'
    rnd_chars = random_string(length=RANDOM_FILE_EXT_LENGTH)
    tmp_sudoers_path = '/tmp/sudoers_{0}'.format(rnd_chars)
    sudoers_entries = list()
    copy_result = execute_command(
        shlex.split(str('{0} cp {1} {2}'.format(sudo_check(), sudoers_path, tmp_sudoers_path))))
    result_message = copy_result[0][1].decode('UTF-8')
    if 'No such file or directory' not in result_message:
        execute_command(shlex.split(str('{0} chmod 755 {1}'.format(sudo_check(), tmp_sudoers_path))))
        with open(tmp_sudoers_path) as tmp_sudoers_file:
            for line in tmp_sudoers_file:
                stripped = line.strip().replace(os.linesep, '')
                if stripped and not stripped.startswith('#'):
                    sudoers_entries.append(stripped)
        execute_command(shlex.split(str('{0} rm {1}'.format(sudo_check(), tmp_sudoers_path))))
    return sudoers_entries