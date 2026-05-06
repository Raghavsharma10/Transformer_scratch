def retrieve(self, namespace, stream, start_time, end_time, start_id,
               configuration, order=ResultOrder.ASCENDING, limit=sys.maxint):
    """
    Retrieves all the events for `stream` from `start_time` (inclusive) till
    `end_time` (inclusive). Alternatively to `start_time`, `start_id` can be
    provided, and then all events from `start_id` (exclusive) till `end_time`
    (inclusive) are returned. `start_id` should be used in cases when the client
    got disconnected from the server before all the events in the requested
    time window had been returned. `order` can be one of ResultOrder.ASCENDING
    or ResultOrder.DESCENDING.

    Returns an iterator over all JSON serialized (strings) events.
    """
    if not start_id:
      start_id = uuid_from_kronos_time(start_time, _type=UUIDType.LOWEST)
    else:
      start_id = TimeUUID(start_id)
    if uuid_to_kronos_time(start_id) > end_time:
      return []
    return self._retrieve(namespace, stream, start_id, end_time, order, limit,
                          configuration)