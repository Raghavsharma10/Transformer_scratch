def _handle_run_exception(self, exc):
        """Process an exception encountered while running the ``run()`` loop.

        Called right before program exits.
        """
        if len(self._current_tups) == 1:
            tup = self._current_tups[0]
            self.raise_exception(exc, tup)
            if self.auto_fail:
                self.fail(tup)