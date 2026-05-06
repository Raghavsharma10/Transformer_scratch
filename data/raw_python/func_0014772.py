def get_server_type():
    """Checks server.ini for server type."""
    server_location_file = os.path.expanduser(SERVER_LOCATION_FILE)
    if not os.path.exists(server_location_file):
        raise Exception(
            "%s not found. Please run 'loom server set "
            "<servertype>' first." % server_location_file)
    config = ConfigParser.SafeConfigParser()
    config.read(server_location_file)
    server_type = config.get('server', 'type')
    return server_type