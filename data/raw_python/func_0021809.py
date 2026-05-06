def register(self, command: str, handler: Any):
        """
        Register a new handler for a specific slash command

        Args:
            command: Slash command
            handler: Callback
        """

        if not command.startswith("/"):
            command = f"/{command}"

        LOG.info("Registering %s to %s", command, handler)
        self._routes[command].append(handler)