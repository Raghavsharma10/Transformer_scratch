def execution_errors(self):
        """
        Return a list of commands that encountered execution errors, with the error.

        Each dictionary entry gives the command dictionary and the error dictionary
        :return: list of commands that gave errors, with their error information
        """
        if self.split_actions:
            # throttling split this action, get errors from the split
            return [dict(e) for s in self.split_actions for e in s.errors]
        else:
            return [dict(e) for e in self.errors]