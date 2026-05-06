def put(self, event_dict, namespace=None):
    """
    Sends a dictionary of `event_dict` of the form {stream_name:
    [event, ...], ...}  to the server.
    """
    # Copy the input, in case we need to modify it by adding a timestamp.
    event_dict = copy.deepcopy(event_dict)

    # Ensure that all events have a timestamp.
    timestamp = kronos_time_now()
    for events in event_dict.itervalues():
      for event in events:
        if TIMESTAMP_FIELD not in event:
          event[TIMESTAMP_FIELD] = timestamp
        else:
          if isinstance(event[TIMESTAMP_FIELD], types.StringTypes):
            event[TIMESTAMP_FIELD] = parse(event[TIMESTAMP_FIELD])
          if isinstance(event[TIMESTAMP_FIELD], datetime):
            event[TIMESTAMP_FIELD] = datetime_to_kronos_time(
              event[TIMESTAMP_FIELD])
        event[LIBRARY_FIELD] = {
          'version': pykronos.__version__,
          'name': 'pykronos'
        }

    namespace = namespace or self.namespace

    if self._blocking:
      return self._put(namespace, event_dict)
    else:
      with self._put_lock:
        self._put_queue.append((namespace, event_dict))