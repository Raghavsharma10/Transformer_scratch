def loop(sock, config=None):

    """Loops over all docker events and executes subscribed callbacks with an
    optional config value.

    :param config: a dictionary with external config values
    """

    if config is None:
        config = {}

    client = docker.Client(base_url=sock)

    # fake a running event for all running containers
    for container in client.containers():
        event_data = {
            'status': "running",
            'id': container['Id'],
            'from': container['Image'],
            'time': container['Created'],
        }

        LOG.debug("incomming event: %s", event_data)

        callbacks = event.filter_callbacks(client, event_data)

        # spawn all callbacks
        gevent.joinall([gevent.spawn(cb, event_data, config) for cb in callbacks])

    # listen for further events
    for raw_data in client.events():

        event_data = json.loads(raw_data)

        LOG.debug("incomming event: %s", event_data)

        callbacks = event.filter_callbacks(client, event_data)

        # spawn all callbacks
        gevent.joinall([gevent.spawn(cb, client, event_data, config) for cb in callbacks])