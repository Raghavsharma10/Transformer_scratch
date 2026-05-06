def parse_known2func(self, args=None, func=None):
        """Parse the command line arguments to the setup function

        This method parses the given command line arguments to the function
        used in the :meth:`setup_args` method to setup up this parser

        Parameters
        ----------
        args: list
            The list of command line arguments
        func: function or str
            An alternative function to use. If None, the last function or the
            one specified through the `setup_as` parameter in the
            :meth:`setup_args` is used.

        Returns
        -------
        object
            What ever is returned by the called function
        list
            The remaining command line arguments that could not be interpreted

        Note
        ----
        This method does not cover subparsers!"""
        ns, remainder = self.parse_known_args(args)
        kws = vars(ns)
        if func is None:
            if self._setup_as:
                func = kws.pop(self._setup_as)
            else:
                func = self._used_functions[-1]
        return func(**kws), remainder