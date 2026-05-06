def _wait_for_read_ready_or_timeout(self, timeout):
        """Returns tuple of whether stdin is ready to read and an event.

        If an event is returned, that event is more pressing than reading
        bytes on stdin to create a keyboard input event.
        If stdin is ready, either there are bytes to read or a SIGTSTP
        triggered by dsusp has been received"""
        remaining_timeout = timeout
        t0 = time.time()
        while True:
            try:
                (rs, _, _) = select.select(
                    [self.in_stream.fileno()] + self.readers,
                    [], [], remaining_timeout)
                if not rs:
                    return False, None
                r = rs[0]  # if there's more than one, get it in the next loop
                if r == self.in_stream.fileno():
                    return True, None
                else:
                    os.read(r, 1024)
                    if self.queued_interrupting_events:
                        return False, self.queued_interrupting_events.pop(0)
                    elif remaining_timeout is not None:
                        remaining_timeout = max(0, t0 + timeout - time.time())
                        continue
                    else:
                        continue

            except select.error:
                if self.sigints:
                    return False, self.sigints.pop()
                if remaining_timeout is not None:
                    remaining_timeout = max(timeout - (time.time() - t0), 0)