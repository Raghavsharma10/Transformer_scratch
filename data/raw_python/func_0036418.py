def apply_config(self, config):
        """
        Sets attributes based on the given config.

        Also adjusts the `results` deque to either expand (padding itself with
        False results) or contract (by removing the oldest results) until it
        matches the required length.
        """
        self.rise = int(config["rise"])
        self.fall = int(config["fall"])

        self.apply_check_config(config)

        if self.results.maxlen == max(self.rise, self.fall):
            return

        results = list(self.results)
        while len(results) > max(self.rise, self.fall):
            results.pop(0)
        while len(results) < max(self.rise, self.fall):
            results.insert(0, False)

        self.results = deque(
            results,
            maxlen=max(self.rise, self.fall)
        )