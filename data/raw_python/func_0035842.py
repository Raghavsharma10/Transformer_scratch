def _retrieve(self, namespace, stream, start_id, end_time, order, limit,
                configuration):
    """
    Yield events from stream starting after the event with id `start_id` until
    and including events with timestamp `end_time`.
    """
    indices = self.index_manager.get_aliases(namespace,
                                             uuid_to_kronos_time(start_id),
                                             end_time)
    if not indices:
      return

    end_id = uuid_from_kronos_time(end_time, _type=UUIDType.HIGHEST)
    end_id.descending = start_id.descending = descending = (
      order == ResultOrder.DESCENDING)

    start_time = uuid_to_kronos_time(start_id)
    body_query = {
      'query': {
        'filtered': {
          'query': {'match_all': {}},
          'filter': {
            'range': {TIMESTAMP_FIELD: {'gte': start_time, 'lte': end_time}}
          }
        }
      }
    }
    order = 'desc' if descending else 'asc'
    sort_query = [
      '%s:%s' % (TIMESTAMP_FIELD, order),
      '%s:%s' % (ID_FIELD, order)
    ]

    last_id = end_id if descending else start_id
    scroll_id = None
    while True:
      size = max(min(limit, configuration['read_size']) / self.shards, 10)
      if scroll_id is None:
        res = self.es.search(index=indices,
                             doc_type=stream,
                             size=size,
                             body=body_query,
                             sort=sort_query,
                             _source=True,
                             scroll='1m',
                             ignore=[400, 404],
                             allow_no_indices=True,
                             ignore_unavailable=True)
      else:
        res = self.es.scroll(scroll_id, scroll='1m')
      if '_scroll_id' not in res:
        break
      scroll_id = res['_scroll_id']
      hits = res.get('hits', {}).get('hits')
      if not hits:
        break

      for hit in hits:
        _id = TimeUUID(hit['_id'], descending=descending)
        if _id <= last_id:
          continue
        last_id = _id
        event = hit['_source']
        del event[LOGSTASH_TIMESTAMP_FIELD]
        yield json.dumps(event)
        limit -= 1
        if limit == 0:
          break

    if scroll_id is not None:
      self.es.clear_scroll(scroll_id)