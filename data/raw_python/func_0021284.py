def delay(self):
        """Sleep for an amount of time to remain under the rate limit."""
        if self.next_request_timestamp is None:
            return
        sleep_seconds = self.next_request_timestamp - time.time()
        if sleep_seconds <= 0:
            return
        message = "Sleeping: {:0.2f} seconds prior to" " call".format(
            sleep_seconds
        )
        log.debug(message)
        time.sleep(sleep_seconds)