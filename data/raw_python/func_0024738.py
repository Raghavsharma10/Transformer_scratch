async def start_timeout(self):
        """Start timeout."""
        self.timeout_handle = self.pyvlx.connection.loop.call_later(
            self.timeout_in_seconds, self.timeout)