def subscribe(self, stream):
        """ Subscribe to a stream.

            :param stream: stream to subscribe to
            :type stream: str
            :raises: :class:`~datasift.exceptions.StreamSubscriberNotStarted`, :class:`~datasift.exceptions.DeleteRequired`, :class:`~datasift.exceptions.StreamNotConnected`

            Used as a decorator, eg.::

                @client.subscribe(stream)
                def subscribe_to_hash(msg):
                    print(msg)
        """
        if not self._stream_process_started:
            raise StreamSubscriberNotStarted()

        def real_decorator(func):
            if not self._on_delete:
                raise DeleteRequired("""An on_delete function is required. You must process delete messages and remove
                 them from your system (if stored) in order to remain compliant with the ToS""")
            if hasattr(self.factory, 'datasift') and 'send_message' in self.factory.datasift:  # pragma: no cover
                self.subscriptions[stream] = func
                self.factory.datasift['send_message'](json.dumps({"action": "subscribe", "hash": stream}).encode("utf8"))
            else:  # pragma: no cover
                raise StreamNotConnected('The client is not connected to DataSift, unable to subscribe to stream')

        return real_decorator