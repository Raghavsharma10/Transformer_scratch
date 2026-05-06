def add(name, default_for_unspecified, total_resource_slots, max_concurrent_sessions,
        max_containers_per_session, max_vfolder_count, max_vfolder_size,
        idle_timeout, allowed_vfolder_hosts):
    '''
    Add a new keypair resource policy.

    NAME: NAME of a new keypair resource policy.
    '''
    with Session() as session:
        try:
            data = session.ResourcePolicy.create(
                name,
                default_for_unspecified=default_for_unspecified,
                total_resource_slots=total_resource_slots,
                max_concurrent_sessions=max_concurrent_sessions,
                max_containers_per_session=max_containers_per_session,
                max_vfolder_count=max_vfolder_count,
                max_vfolder_size=max_vfolder_size,
                idle_timeout=idle_timeout,
                allowed_vfolder_hosts=allowed_vfolder_hosts,
            )
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if not data['ok']:
            print_fail('KeyPair Resource Policy creation has failed: {0}'
                       .format(data['msg']))
            sys.exit(1)
        item = data['resource_policy']
        print('Keypair resource policy ' + item['name'] + ' is created.')