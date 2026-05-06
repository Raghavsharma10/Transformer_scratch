async def loop(self):
        """Pulse every timeout seconds until stopped."""
        while not self.stopped:
            self.timeout_handle = self.pyvlx.connection.loop.call_later(
                self.timeout_in_seconds, self.loop_timeout)
            await self.loop_event.wait()
            if not self.stopped:
                self.loop_event.clear()
                await self.pulse()
        self.cancel_loop_timeout()
        self.stopped_event.set()