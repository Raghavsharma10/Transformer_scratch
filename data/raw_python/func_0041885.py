def get_subparser(self, name):
        """
        Convenience method to get a certain subparser

        Parameters
        ----------
        name: str
            The name of the subparser

        Returns
        -------
        FuncArgParser
            The subparsers corresponding to `name`
        """
        if self._subparsers_action is None:
            raise ValueError("%s has no subparsers defined!" % self)
        return self._subparsers_action.choices[name]