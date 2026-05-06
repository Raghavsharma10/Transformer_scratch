def _process(self, on_commit: UpdateCallable, on_rollback: UpdateCallable) -> Any:
        """
        Process action. oncommit is a callback to execute action, onrollback is
        a callback to execute if the oncommit() has been called and a rollback
        is required
        """

        _debug("---> commiting", on_commit)
        result = self._do_with_retry(on_commit)

        if len(self._transactions) > 0:
            # add statement to rollback log in case something goes wrong
            self._transactions[-1].insert(0, on_rollback)

        return result