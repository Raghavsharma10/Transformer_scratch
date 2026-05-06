def write_sudoers_entry(username=None, sudoers_entry=None):
    """Write sudoers entry.

    args:
        user (User): Instance of User containing sudoers entry.

    returns:
        str: sudoers entry for the specified user.
    """

    sudoers_path = '/etc/sudoers'
    rnd_chars = random_string(length=RANDOM_FILE_EXT_LENGTH)
    tmp_sudoers_path = '/tmp/sudoers_{0}'.format(rnd_chars)
    execute_command(
        shlex.split(str('{0} cp {1} {2}'.format(sudo_check(), sudoers_path, tmp_sudoers_path))))
    execute_command(
        shlex.split(str('{0} chmod 777 {1}'.format(sudo_check(), tmp_sudoers_path))))
    with open(tmp_sudoers_path, mode=text_type('r')) as tmp_sudoers_file:
        sudoers_entries = tmp_sudoers_file.readlines()
    sudoers_output = list()
    for entry in sudoers_entries:
        if entry and not entry.startswith(username):
            sudoers_output.append(entry)
    if sudoers_entry:
        sudoers_output.append('{0} {1}'.format(username, sudoers_entry))
        sudoers_output.append('\n')
    with open(tmp_sudoers_path, mode=text_type('w+')) as tmp_sudoers_file:
        tmp_sudoers_file.writelines(sudoers_output)
    sudoers_check_result = execute_command(
        shlex.split(str('{0} {1} -cf {2}'.format(sudo_check(), LINUX_CMD_VISUDO, tmp_sudoers_path))))
    if sudoers_check_result[1] > 0:
        raise ValueError(sudoers_check_result[0][1])
    execute_command(
        shlex.split(str('{0} cp {1} {2}'.format(sudo_check(), tmp_sudoers_path, sudoers_path))))
    execute_command(shlex.split(str('{0} chown root:root {1}'.format(sudo_check(), sudoers_path))))
    execute_command(shlex.split(str('{0} chmod 440 {1}'.format(sudo_check(), sudoers_path))))
    execute_command(shlex.split(str('{0} rm {1}'.format(sudo_check(), tmp_sudoers_path))))