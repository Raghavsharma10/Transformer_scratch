def _handle_sigint(self, signum: int, frame: Any) -> None:
        """Shutdown after processing current task."""
        logger.warning("Catched SIGINT")
        self.shutdown()