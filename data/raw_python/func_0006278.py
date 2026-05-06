def put_event(self, data):
        """ Create event from `data` and add it to the queue.

        :arg json data: The JSON data from which to create the event.

        :Raises: :class:`pygerrit.error.GerritError` if the queue is full, or
            the factory could not create the event.

        """
        try:
            event = self._factory.create(data)
            self._events.put(event)
        except Full:
            raise GerritError("Unable to add event: queue is full")