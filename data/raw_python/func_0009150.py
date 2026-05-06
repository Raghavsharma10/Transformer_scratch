def _handle_run_exception(self, exc):
        """Process an exception encountered while running the ``run()`` loop.

        Called right before program exits.
        """
        self.raise_exception(exc, self._current_tups)

        if self.auto_fail:
            failed = set()
            for key, batch in iteritems(self._batches):
                # Only wipe out batches other than current for exit_on_exception
                if self.exit_on_exception or key == self._current_key:
                    for tup in batch:
                        self.fail(tup)
                        failed.add(tup.id)

            # Fail current batch or tick Tuple if we have one
            for tup in self._current_tups:
                if tup.id not in failed:
                    self.fail(tup)

            # Reset current batch info
            self._batches[self._current_key] = []
            self._current_key = None