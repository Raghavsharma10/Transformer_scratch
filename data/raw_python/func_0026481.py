def drop_privileges(uid_name='hfos', gid_name='hfos'):
    """Attempt to drop privileges and change user to 'hfos' user/group"""

    if os.getuid() != 0:
        hfoslog("Not root, cannot drop privileges", lvl=warn, emitter='CORE')
        return

    try:
        # Get the uid/gid from the name
        running_uid = pwd.getpwnam(uid_name).pw_uid
        running_gid = grp.getgrnam(gid_name).gr_gid

        # Remove group privileges
        os.setgroups([])

        # Try setting the new uid/gid
        os.setgid(running_gid)
        os.setuid(running_uid)

        # Ensure a very conservative umask
        # old_umask = os.umask(22)
        hfoslog('Privileges dropped', emitter='CORE')
    except Exception as e:
        hfoslog('Could not drop privileges:', e, type(e), exc=True, lvl=error, emitter='CORE')