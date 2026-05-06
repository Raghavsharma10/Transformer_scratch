def parse_ssh_destination(destination):
    """Parses the SSH destination argument.
    """
    match = _re_ssh.match(destination)
    if not match:
        raise InvalidDestination("Invalid destination: %s" % destination)
    user, password, host, port = match.groups()
    info = {}
    if user:
        info['username'] = user
    else:
        info['username'] = getpass.getuser()
    if password:
        info['password'] = password
    if port:
        info['port'] = int(port)
    info['hostname'] = host

    return info