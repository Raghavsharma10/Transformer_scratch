def run(self):
        """
        Calls the `perform()` method defined by subclasses and stores the
        result in a `results` deque.

        After the result is determined the `results` deque is analyzed to see
        if the `passing` flag should be updated.  If the check was considered
        passing and the previous `self.fall` number of checks failed, the check
        is updated to not be passing.  If the check was not passing and the
        previous `self.rise` number of checks passed, the check is updated to
        be considered passing.
        """
        logger.debug("Running %s check", self.name)

        try:
            result = self.perform()
        except Exception:
            logger.exception("Error while performing %s check", self.name)
            result = False

        logger.debug("Result: %s", result)

        self.results.append(result)
        if self.passing and not any(self.last_n_results(self.fall)):
            logger.info(
                "%s check failed %d time(s), no longer passing.",
                self.name, self.fall,
            )
            self.passing = False
        if not self.passing and all(self.last_n_results(self.rise)):
            logger.info(
                "%s check passed %d time(s), is now passing.",
                self.name, self.rise
            )
            self.passing = True