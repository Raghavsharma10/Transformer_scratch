def ssh_client(host, username=None, password=None, **kwargs):
    """
    Create a new :class:`paramiko.SSHClient`, connect it and return the
    instance.

    This is a simple wrapper around the connect method that add some good
    defaults when using username/password to connect.
    """
    client = paramiko.SSHClient()
    # save hostname and username on the instance - this is a ugly hack
    # but I don't see any other way to do that for now. Note that
    # this is only used for SshSession.__str__.
    client.hostname = host
    client.username = username
    if username is not None:
        kwargs['username'] = username
    if password is not None:
        kwargs['password'] = password
    if username is not None and password is not None:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs.setdefault('look_for_keys', False)
    client.connect(host, **kwargs)
    return client