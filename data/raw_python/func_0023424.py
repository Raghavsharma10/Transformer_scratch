def _process_pending_variables(self):
        """ Try to apply the variables that were set but not known yet.
        """
        # Clear our list of pending variables
        self._pending_variables, pending = {}, self._pending_variables
        # Try to apply it. On failure, it will be added again
        for name, data in pending.items():
            self[name] = data