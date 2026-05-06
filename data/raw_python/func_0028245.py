def resource_policies(ctx):
    '''
    List and manage resource policies.
    (admin privilege required)
    '''
    if ctx.invoked_subcommand is not None:
        return
    fields = [
        ('Name', 'name'),
        ('Created At', 'created_at'),
        ('Default for Unspecified', 'default_for_unspecified'),
        ('Total Resource Slot', 'total_resource_slots'),
        ('Max Concurrent Sessions', 'max_concurrent_sessions'),
        ('Max Containers per Session', 'max_containers_per_session'),
        ('Max vFolder Count', 'max_vfolder_count'),
        ('Max vFolder Size', 'max_vfolder_size'),
        ('Idle Timeeout', 'idle_timeout'),
        ('Allowed vFolder Hosts', 'allowed_vfolder_hosts'),
    ]
    with Session() as session:
        try:
            items = session.ResourcePolicy.list(fields=(item[1] for item in fields))
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if len(items) == 0:
            print('There are no keypair resource policies.')
            return
        print(tabulate((item.values() for item in items),
                       headers=(item[0] for item in fields)))