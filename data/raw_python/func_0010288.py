def streams(self):
        """Property providing access to the :class:`.StreamsAPI`"""
        if self._streams_api is None:
            self._streams_api = self.get_streams_api()
        return self._streams_api