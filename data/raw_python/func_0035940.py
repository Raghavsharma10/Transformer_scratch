def _insert(self, namespace, stream, events, configuration):
    """
    `stream` is the name of a stream and `events` is a list of
    (TimeUUID, event) to insert. Make room for the events to insert if
    necessary by deleting the oldest events. Then insert each event in time
    sorted order.
    """
    max_items = configuration['max_items']
    for _id, event in events:
      while len(self.db[namespace][stream]) >= max_items:
        self.db[namespace][stream].pop(0)
      bisect.insort(self.db[namespace][stream], Event(_id, event))