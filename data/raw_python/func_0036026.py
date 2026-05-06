def delete(self, stream, start_time, end_time, start_id=None, namespace=None):
    """
    Delete events in the stream with name `stream` that occurred between
    `start_time` and `end_time` (both inclusive).  An optional `start_id` allows
    the client to delete events starting from after an ID rather than starting
    at a timestamp.
    """
    if isinstance(start_time, types.StringTypes):
      start_time = parse(start_time)
    if isinstance(end_time, types.StringTypes):
      end_time = parse(end_time)
    if isinstance(start_time, datetime):
      start_time = datetime_to_kronos_time(start_time)
    if isinstance(end_time, datetime):
      end_time = datetime_to_kronos_time(end_time)
    request_dict = {
      'stream': stream,
      'end_time': end_time
    }
    if start_id:
      request_dict['start_id'] = start_id
    else:
      request_dict['start_time'] = start_time

    namespace = namespace or self.namespace
    if namespace is not None:
      request_dict['namespace'] = namespace

    return self._make_request(self._delete_url, data=request_dict)