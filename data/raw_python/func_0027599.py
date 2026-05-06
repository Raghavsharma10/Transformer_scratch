def run_heartbeat_continuously(self) -> threading.Event:
        """
        For a long runing handler, there is a danger that we do not send a heartbeat message or activity on the
        connection whilst we are running the handler. With a default heartbeat of 30s, for example, there is a risk
        that a handler which takes more than 15s will fail to send the heartbeat in time and then the broker will
        reset the connection. So we spin up another thread, where the user has marked the thread as having a
        long-running thread
        :return: an event to cancel the thread
        """

        cancellation_event = threading.Event()

        # Effectively a no-op if we are not actually a long-running thread
        if not self._is_long_running_handler:
            return cancellation_event

        self._logger.debug("Running long running handler on %s", self._conn)

        def _send_heartbeat(cnx: BrokerConnection, period: int, logger: logging.Logger) -> None:
                while not cancellation_event.is_set():
                    cnx.heartbeat_check()
                    time.sleep(period)
                logger.debug("Signalled to exit long-running handler heartbeat")


        heartbeat_thread = threading.Thread(target=_send_heartbeat, args=(self._conn, 1, self._logger), daemon=True)
        self._logger.debug("Begin heartbeat thread for  %s", self._conn)
        heartbeat_thread.start()
        self._logger.debug("Heartbeat running on thread for  %s", self._conn)
        return cancellation_event