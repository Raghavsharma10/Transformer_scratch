def write(self, data):
        """Buffers some data to be sent to the host:port in a non blocking way.

        So the data is always buffered and not sent on the socket in a
        synchronous way.

        You can give a WriteBuffer as parameter. The internal Connection
        WriteBuffer will be extended with this one (without copying).

        Args:
            data (str or WriteBuffer): string (or WriteBuffer) to write to
                the host:port.
        """
        if isinstance(data, WriteBuffer):
            self._write_buffer.append(data)
        else:
            if len(data) > 0:
                self._write_buffer.append(data)
        if self.aggressive_write:
            self._handle_write()
        if self._write_buffer._total_length > 0:
            self._register_or_update_event_handler(write=True)