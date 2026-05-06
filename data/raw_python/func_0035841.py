def _insert(self, namespace, stream, events, configuration):
    """
    `namespace` acts as db for different streams
    `stream` is the name of a stream and `events` is a list of events to
    insert.
    """
    index = self.index_manager.get_index(namespace)
    start_dts_to_add = set()

    def actions():
      for _id, event in events:
        dt = kronos_time_to_datetime(uuid_to_kronos_time(_id))
        start_dts_to_add.add(_round_datetime_down(dt))
        event['_index'] = index
        event['_type'] = stream
        event[LOGSTASH_TIMESTAMP_FIELD] = dt.isoformat()

        yield event

    list(es_helpers.streaming_bulk(self.es, actions(), chunk_size=1000,
                                   refresh=self.force_refresh))
    self.index_manager.add_aliases(namespace,
                                   index,
                                   start_dts_to_add)