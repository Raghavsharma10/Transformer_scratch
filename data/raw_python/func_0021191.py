def execute_command(self, *args, **kwargs):
        """Execute a command on the connected server."""
        try:
            return self.get_connection().execute_command(*args, **kwargs)
        except ConnectionError as e:
            logger.warn('trying to reconnect')
            self.connect()
            logger.warn('connected')
            raise