def _retrieve(self, namespace, stream, start_id, end_time, order, limit,
                configuration):
    """
    Retrieve events for `stream` between `start_id` and `end_time`.
    `stream` : The stream to return events for.
    `start_id` : Return events with id > `start_id`.
    `end_time` : Return events ending <= `end_time`.
    `order` : Whether to return the results in ResultOrder.ASCENDING
              or ResultOrder.DESCENDING time-order.
    `configuration` : A dictionary of settings to override any default
                      settings, such as number of shards or width of a
                      time interval.
    """
    stream = self.get_stream(namespace, stream, configuration)
    events = stream.iterator(start_id,
                             uuid_from_kronos_time(end_time,
                                                   _type=UUIDType.HIGHEST),
                             order == ResultOrder.DESCENDING, limit)
    events = events.__iter__()
    event = events.next()
    # If first event's ID is equal to `start_id`, skip it.
    if event.id != start_id:
      yield event.json
    while True:
      yield events.next().json