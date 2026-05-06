def spawn_managed_host(config_file, manager, connect_on_start=True):
    """
    Spawns a managed host, if it is not already running
    """

    data = manager.request_host_status(config_file)

    is_running = data['started']

    # Managed hosts run as persistent processes, so it may already be running
    if is_running:
        host_status = json.loads(data['host']['output'])
        logfile = data['host']['logfile']
    else:
        data = manager.start_host(config_file)
        host_status = json.loads(data['output'])
        logfile = data['logfile']

    host = JSHost(
        status=host_status,
        logfile=logfile,
        config_file=config_file,
        manager=manager
    )

    if not is_running and settings.VERBOSITY >= verbosity.PROCESS_START:
        print('Started {}'.format(host.get_name()))

    if connect_on_start:
        host.connect()

    return host