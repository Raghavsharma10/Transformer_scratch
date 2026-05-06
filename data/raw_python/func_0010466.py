def run(self) -> None:
        """Runs the worker and consumes messages from RabbitMQ.
        Returns only after `shutdown()` is called.

        """
        if self._logging_level:
            logging.basicConfig(
                level=getattr(logging, self._logging_level.upper()),
                format="%(levelname).1s %(name)s.%(funcName)s:%(lineno)d - %(message)s")

        signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        if platform.system() != 'Windows':
            # These features will not be available on Windows, but that is OK.
            # Read this issue for more details:
            # https://github.com/cenkalti/kuyruk/issues/54
            signal.signal(signal.SIGHUP, self._handle_sighup)
            signal.signal(signal.SIGUSR1, self._handle_sigusr1)
            signal.signal(signal.SIGUSR2, self._handle_sigusr2)

        self._started_at = os.times().elapsed

        for t in self._threads:
            t.start()

        try:
            signals.worker_start.send(self.kuyruk, worker=self)
            self._consume_messages()
            signals.worker_shutdown.send(self.kuyruk, worker=self)
        finally:
            self.shutdown_pending.set()
            for t in self._threads:
                t.join()

        logger.debug("End run worker")