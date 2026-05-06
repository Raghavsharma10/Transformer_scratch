def move_to(self, position):
        """Set the Coordinator to a specific endpoint or time, or load state from a token.

        :param position: "trim_horizon", "latest", :class:`~datetime.datetime`, or a
            :attr:`Coordinator.token <bloop.stream.coordinator.Coordinator.token>`
        """
        if isinstance(position, collections.abc.Mapping):
            move = _move_stream_token
        elif hasattr(position, "timestamp") and callable(position.timestamp):
            move = _move_stream_time
        elif isinstance(position, str) and position.lower() in ["latest", "trim_horizon"]:
            move = _move_stream_endpoint
        else:
            raise InvalidPosition("Don't know how to move to position {!r}".format(position))
        move(self, position)