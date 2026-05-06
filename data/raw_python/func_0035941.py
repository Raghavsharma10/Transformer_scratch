def _delete(self, namespace, stream, start_id, end_time, configuration):
    """
    Delete events with id > `start_id` and end_time <= `end_time`.
    """
    start_id_event = Event(start_id)
    end_id_event = Event(uuid_from_kronos_time(end_time,
                                               _type=UUIDType.HIGHEST))
    stream_events = self.db[namespace][stream]

    # Find the interval our events belong to.
    lo = bisect.bisect_left(stream_events, start_id_event)
    if lo + 1 > len(stream_events):
      return 0, []
    if stream_events[lo] == start_id_event:
      lo += 1
    hi = bisect.bisect_right(stream_events, end_id_event)

    del stream_events[lo:hi]
    return max(0, hi - lo), []