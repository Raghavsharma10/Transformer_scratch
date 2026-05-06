def get_results(self):
        """Run the linter, parse, and return result list.

        If a linter specified by the user is not found, return an error message
        as result.
        """
        try:
            stdout, stderr = self._lint()
            # Can't return a generator from a subprocess
            return list(stdout), stderr or []
        except FileNotFoundError as exception:
            # Error if the linter was not found but was chosen by the user
            if self._linter.name in self.config.user_linters:
                error_msg = 'Could not find {}. Did you install it? ' \
                    'Got exception: {}'.format(self._linter.name, exception)
                return [[], [error_msg]]
            # If the linter was not chosen by the user, do nothing
            return [[], []]