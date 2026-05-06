def sleep(self):
        """Wait for the sleep time of the last response, to avoid being rate
        limited."""

        if self.next_time and time.time() < self.next_time:
            time.sleep(self.next_time - time.time())