def _release_command_buffer(self, command_buffer):
        """This is called by the command buffer when it closes."""
        if command_buffer.closed:
            return

        self._cb_poll.unregister(command_buffer.host_id)
        self.connection_pool.release(command_buffer.connection)
        command_buffer.connection = None