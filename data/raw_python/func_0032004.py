def get_send_enable(self):
        """
        Return true if sending of sessions is enabled on the watch
        """

        # We have to open this queue before we make the request, to ensure we don't miss the response.
        queue = self._pebble.get_endpoint_queue(DataLogging)

        self._pebble.send_packet(DataLogging(data=DataLoggingGetSendEnableRequest()))
        enabled = False
        while True:
            result = queue.get().data
            if isinstance(result, DataLoggingGetSendEnableResponse):
                enabled = result.enabled
                break

        queue.close()
        return enabled