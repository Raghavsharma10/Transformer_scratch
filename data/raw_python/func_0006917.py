def apt_add_repository_from_apt_string(apt_string, apt_file):
    """ adds a new repository file for apt """

    apt_file_path = '/etc/apt/sources.list.d/%s' % apt_file

    if not file_contains(apt_file_path, apt_string.lower(), use_sudo=True):
        file_append(apt_file_path, apt_string.lower(), use_sudo=True)

        with hide('running', 'stdout'):
            sudo("DEBIAN_FRONTEND=noninteractive apt-get update")