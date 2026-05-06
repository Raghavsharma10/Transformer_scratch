def dir_ensure(location, recursive=False, mode=None,
               owner=None, group=None, use_sudo=False):
    """ cuisine dir_ensure doesn't do sudo, so we implement our own
    Ensures that there is a remote directory at the given location,
    optionally updating its mode/owner/group.
    If we are not updating the owner/group then this can be done as a single
    ssh call, so use that method, otherwise set owner/group after creation."""

    args = ''
    if recursive:
        args = args + ' -p '

    if not dir_exists(location):
        if use_sudo:
            sudo('mkdir %s %s' % (args, location))
        else:
            run('mkdir %s %s' % (args, location))

    if owner or group or mode:
        if use_sudo:
            dir_attribs(location,
                        owner=owner,
                        group=group,
                        mode=mode,
                        recursive=recursive,
                        use_sudo=True)
        else:
            dir_attribs(location,
                        owner=owner,
                        group=group,
                        mode=mode,
                        recursive=recursive)
    return True