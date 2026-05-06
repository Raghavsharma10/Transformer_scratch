def parse_username_password_hostname(remote_url):
    """
    Parse a command line string and return username, password, remote hostname and remote path.

    :param remote_url: A command line string.
    :return: A tuple, containing username, password, remote hostname and remote path.
    """
    assert remote_url
    assert ':' in remote_url

    if '@' in remote_url:
        username, hostname = remote_url.rsplit('@', 1)
    else:
        username, hostname = None, remote_url

    hostname, remote_path = hostname.split(':', 1)

    password = None
    if username and ':' in username:
        username, password = username.split(':', 1)

    assert hostname
    assert remote_path
    return username, password, hostname, remote_path