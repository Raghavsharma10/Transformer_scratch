def poll(connection: connection, timeout: float=1.0) -> Iterable[Event]:
    """Poll the connection for notification events.

    This method operates as an iterable. It will keep returning events until
    all events have been read.

    Parameters
    ----------
    connection: psycopg2.extensions.connection
        Active connection to a PostGreSQL database.
    timeout: float
        Number of seconds to block for an event before timing out.

    Returns
    -------
    event: Event or None
        If an event is available, an Event is returned.
        If no event is available, None is returned.

    Examples
    --------
    >>> events = [evt for evt in connection.poll()]

    >>> for evt in connection.poll():
            print(evt)

    """

    if timeout > 0.0:
        log('Polling for events (Blocking, {} seconds)...'.format(timeout), logger_name=_LOGGER_NAME)
    else:
        log('Polling for events (Non-Blocking)...', logger_name=_LOGGER_NAME)
    if select.select([connection], [], [], timeout) == ([], [], []):
        log('...No events found', logger_name=_LOGGER_NAME)
        return
    else:
        log('Events', logger_name=_LOGGER_NAME)
        log('------', logger_name=_LOGGER_NAME)
        connection.poll()
        while connection.notifies:
            event = connection.notifies.pop(0)
            log(str(event), logger_name=_LOGGER_NAME)
            yield Event.fromjson(event.payload)