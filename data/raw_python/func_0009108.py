def rsync(*args, **kwargs):
    """ wrapper around the rsync command.

        the ssh connection arguments are set automatically.

        any args are just passed directly to rsync.
        you can use {host_string} in place of the server.

        the kwargs are passed on the 'local' fabric command.
        if not set, 'capture' is set to False.

        example usage:
        rsync('-pthrvz', "{host_string}:/some/src/directory", "some/destination/")
    """
    kwargs.setdefault('capture', False)
    replacements = dict(
        host_string="{user}@{host}".format(
            user=env.instance.config.get('user', 'root'),
            host=env.instance.config.get(
                'host', env.instance.config.get(
                    'ip', env.instance.uid))))
    args = [x.format(**replacements) for x in args]
    ssh_info = env.instance.init_ssh_key()
    ssh_info.pop('host')
    ssh_info.pop('user')
    ssh_args = env.instance.ssh_args_from_info(ssh_info)
    cmd_parts = ['rsync']
    cmd_parts.extend(['-e', "ssh %s" % shjoin(ssh_args)])
    cmd_parts.extend(args)
    cmd = shjoin(cmd_parts)
    return local(cmd, **kwargs)