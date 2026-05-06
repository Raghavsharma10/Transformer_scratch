def get_hostmap_deprecated(profiles):
    '''
    This method mangles combine_profiles and get_hostmap, and is deprecated.  At
    this point it only returns the hostmap
    '''

    hostmap = dict()  # map pilot IDs to host names
    for pname, prof in profiles.iteritems():

        if not len(prof):
            continue

        if not prof[0][ru.MSG]:
            continue

        host, ip, _, _, _ = prof[0][ru.MSG].split(':')
        host_id = '%s:%s' % (host, ip)

        for row in prof:

            if 'agent_0.prof' in pname    and \
                    row[ru.EVENT] == 'advance' and \
                    row[ru.STATE] == rps.PMGR_ACTIVE:
                hostmap[row[ru.UID]] = host_id
                break

    return hostmap