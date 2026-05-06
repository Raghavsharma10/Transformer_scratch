def print_status(self):
        """Print out the current tweet rate and reset the counter"""
        tweets = self.received
        now = time.time()
        diff = now - self.since
        self.since = now
        self.received = 0
        if diff > 0:
            logger.info("Receiving tweets at %s tps", tweets / diff)