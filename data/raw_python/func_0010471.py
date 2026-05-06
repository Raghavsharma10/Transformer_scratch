def _handle_sighup(self, signum: int, frame: Any) -> None:
        """Used internally to fail the task when connection to RabbitMQ is
        lost during the execution of the task.

        """
        logger.warning("Catched SIGHUP")
        exc_info = self._heartbeat_exc_info
        self._heartbeat_exc_info = None
        # Format exception info to see in tools like Sentry.
        formatted_exception = ''.join(traceback.format_exception(*exc_info))  # noqa
        raise HeartbeatError(exc_info)