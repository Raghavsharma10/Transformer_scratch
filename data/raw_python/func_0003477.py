def _common_setup(self):
        """
        The minimal amount of setup done by both setup() and no_setup().
        """
        self._started = True
        self._reactor = self._reactorFactory()
        self._registry = ResultRegistry()
        # We want to unblock EventualResult regardless of how the reactor is
        # run, so we always register this:
        self._reactor.addSystemEventTrigger(
            "before", "shutdown", self._registry.stop)