async def do_api_call(self):
        """Start. Sending and waiting for answer."""
        self.pyvlx.connection.register_frame_received_cb(
            self.response_rec_callback)
        await self.send_frame()
        await self.start_timeout()
        await self.response_received_or_timeout.wait()
        await self.stop_timeout()
        self.pyvlx.connection.unregister_frame_received_cb(self.response_rec_callback)