def list(self):
        """
        List all available data logging sessions
        """

        # We have to open this queue before we make the request, to ensure we don't miss the response.
        queue = self._pebble.get_endpoint_queue(DataLogging)

        self._pebble.send_packet(DataLogging(data=DataLoggingReportOpenSessions(sessions=[])))

        sessions = []
        while True:
            try:
                result = queue.get(timeout=2).data
            except TimeoutError:
                break
            if isinstance(result, DataLoggingDespoolOpenSession):
                self._pebble.send_packet(DataLogging(data=DataLoggingACK(
                                                     session_id=result.session_id)))
                sessions.append(result.__dict__)

        queue.close()
        return sessions