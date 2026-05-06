def join(self, timeout=None):
        """Waits for all outstanding responses to come back or the timeout
        to be hit.
        """
        remaining = timeout

        while self._cb_poll and (remaining is None or remaining > 0):
            now = time.time()
            rv = self._cb_poll.poll(remaining)
            if remaining is not None:
                remaining -= (time.time() - now)

            for command_buffer, event in rv:
                # This command buffer still has pending requests which
                # means we have to send them out first before we can read
                # all the data from it.
                if command_buffer.has_pending_requests:
                    if event == 'close':
                        self._try_reconnect(command_buffer)
                    elif event == 'write':
                        self._send_or_reconnect(command_buffer)

                # The general assumption is that all response is available
                # or this might block.  On reading we do not use async
                # receiving.  This generally works because latency in the
                # network is low and redis is super quick in sending.  It
                # does not make a lot of sense to complicate things here.
                elif event in ('read', 'close'):
                    try:
                        command_buffer.wait_for_responses(self)
                    finally:
                        self._release_command_buffer(command_buffer)

        if self._cb_poll and timeout is not None:
            raise TimeoutError('Did not receive all data in time.')