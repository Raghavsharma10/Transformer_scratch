def _shutdown_timer(self) -> None:
        """Counts down from MAX_WORKER_RUN_TIME. When it reaches zero sutdown
        gracefully.

        """
        remaining = self._max_run_time - self.uptime
        if not self.shutdown_pending.wait(remaining):
            logger.warning('Run time reached zero')
            self.shutdown()