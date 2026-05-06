def dir_attribs(location, mode=None, owner=None,
                group=None, recursive=False, use_sudo=False):
    """ cuisine dir_attribs doesn't do sudo, so we implement our own
        Updates the mode/owner/group for the given remote directory."""
    args = ''
    if recursive:
        args = args + ' -R '

    if mode:
        if use_sudo:
            sudo('chmod %s %s %s' % (args, mode,  location))
        else:
            run('chmod %s %s %s' % (args, mode,  location))
    if owner:
        if use_sudo:
            sudo('chown %s %s %s' % (args, owner, location))
        else:
            run('chown %s %s %s' % (args, owner, location))
    if group:
        if use_sudo:
            sudo('chgrp %s %s %s' % (args, group, location))
        else:
            run('chgrp %s %s %s' % (args, group, location))
    return True