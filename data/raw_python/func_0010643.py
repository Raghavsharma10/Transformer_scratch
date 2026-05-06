def check_missing(self, args):
        """ Returns the names of all options that are required but were not specified.

        All options that don't have a default value are required in order to run the
        workflow.

        Args:
            args (dict): A dictionary of the provided arguments that is checked for
                         missing options.

        Returns:
            list: A list with the names of the options that are missing from the
                  provided arguments.
        """
        return [opt.name for opt in self
                if (opt.name not in args) and (opt.default is None)]