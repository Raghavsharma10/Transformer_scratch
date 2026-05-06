def _handle_sigusr2(self, signum: int, frame: Any) -> None:
        """Drop current task."""
        logger.warning("Catched SIGUSR2")
        if self.current_task:
            logger.warning("Dropping current task...")
            raise Discard