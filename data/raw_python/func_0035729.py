def _delete(self, namespace, stream, start_id, end_time, configuration):
    """
    Delete events for `stream` between `start_id` and `end_time`.
    `stream` : The stream to delete events for.
    `start_id` : Delete events with id > `start_id`.
    `end_time` : Delete events ending <= `end_time`.
    `configuration` : A dictionary of settings to override any default
                      settings, such as number of shards or width of a
                      time interval.
    """
    stream = self.get_stream(namespace, stream, configuration)
    return stream.delete(start_id,
                         uuid_from_kronos_time(end_time,
                                               _type=UUIDType.HIGHEST))