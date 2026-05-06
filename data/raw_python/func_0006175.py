def _get_command_buffer(self, host_id, command_name):
        """Returns the command buffer for the given command and arguments."""
        buf = self._cb_poll.get(host_id)
        if buf is not None:
            return buf

        if self._max_concurrency is not None:
            while len(self._cb_poll) >= self._max_concurrency:
                self.join(timeout=1.0)

        def connect():
            return self.connection_pool.get_connection(
                command_name, shard_hint=host_id)
        buf = CommandBuffer(host_id, connect, self.auto_batch)
        self._cb_poll.register(host_id, buf)
        return buf