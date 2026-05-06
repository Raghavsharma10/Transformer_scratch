def close(self):
        """Close any open window.

        Note that this only works with non-blocking methods.

        """
        if self._process:
            # Be nice first.
            self._process.send_signal(signal.SIGINT)

            # If it doesn't close itself promptly, be brutal.
            # Python 3.2+ added the timeout option to wait() and the
            # corresponding TimeoutExpired exception. If they exist, use them.
            if hasattr(subprocess, 'TimeoutExpired'):
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._process.send_signal(signal.SIGKILL)

            # Otherwise, roll our own polling loop.
            else:
                # Give it 1s, checking every 10ms.
                count = 0
                while count < 100:
                    if self._process.poll() is not None:
                        break
                    time.sleep(0.01)

                # Still hasn't quit.
                if self._process.poll() is None:
                    self._process.send_signal(signal.SIGKILL)

            # Clean up.
            self._process = None