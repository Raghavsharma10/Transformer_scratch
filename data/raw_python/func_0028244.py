def resource_policy(name):
    """
    Show details about a keypair resource policy. When `name` option is omitted, the
    resource policy for the current access_key will be returned.
    """
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
            rp = session.ResourcePolicy(session.config.access_key)
            info = rp.info(name, fields=(item[1] for item in fields))
        except Exception as e:
            print_error(e)
            sys.exit(1)
        rows = []
        if info is None:
            print('No such resource policy.')
            sys.exit(1)
        for name, key in fields:
            rows.append((name, info[key]))
        print(tabulate(rows, headers=('Field', 'Value')))