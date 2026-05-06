def update(self, response_headers):
        """Update the state of the rate limiter based on the response headers.

        This method should only be called following a HTTP request to reddit.

        Response headers that do not contain x-ratelimit fields will be treated
        as a single request. This behavior is to error on the safe-side as such
        responses should trigger exceptions that indicate invalid behavior.

        """
        if "x-ratelimit-remaining" not in response_headers:
            if self.remaining is not None:
                self.remaining -= 1
                self.used += 1
            return

        now = time.time()
        prev_remaining = self.remaining

        seconds_to_reset = int(response_headers["x-ratelimit-reset"])
        self.remaining = float(response_headers["x-ratelimit-remaining"])
        self.used = int(response_headers["x-ratelimit-used"])
        self.reset_timestamp = now + seconds_to_reset

        if self.remaining <= 0:
            self.next_request_timestamp = self.reset_timestamp
            return

        if prev_remaining is not None and prev_remaining > self.remaining:
            estimated_clients = prev_remaining - self.remaining
        else:
            estimated_clients = 1.0

        self.next_request_timestamp = min(
            self.reset_timestamp,
            now + (estimated_clients * seconds_to_reset / self.remaining),
        )