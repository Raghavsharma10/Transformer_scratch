def get_hostmap(profile):
    '''
    We abuse the profile combination to also derive a pilot-host map, which
    will tell us on what exact host each pilot has been running.  To do so, we
    check for the PMGR_ACTIVE advance event in agent_0.prof, and use the NTP
    sync info to associate a hostname.
    '''
    # FIXME: This should be replaced by proper hostname logging
    #        in `pilot.resource_details`.

    hostmap = dict()  # map pilot IDs to host names
    for entry in profile:
        if entry[ru.EVENT] == 'hostname':
            hostmap[entry[ru.UID]] = entry[ru.MSG]

    return hostmap