async def send_frame(self, frame):
        """Send frame to API via connection."""
        if not self.connection.connected:
            await self.connect()
            await self.update_version()
            await set_utc(pyvlx=self)
            await house_status_monitor_enable(pyvlx=self)
        self.connection.write(frame)