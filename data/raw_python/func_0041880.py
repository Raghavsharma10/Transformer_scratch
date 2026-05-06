def parse2func(self, args=None, func=None):
        """Parse the command line arguments to the setup function

        This method parses the given command line arguments to the function
        used in the :meth:`setup_args` method to setup up this parser

        Parameters
        ----------
        args: list
            The list of command line arguments
        func: function
            An alternative function to use. If None, the last function or the
            one specified through the `setup_as` parameter in the
            :meth:`setup_args` is used.

        Returns
        -------
        object
            What ever is returned by the called function

        Note
        ----
        This method does not cover subparsers!"""
        kws = vars(self.parse_args(args))
        if func is None:
            if self._setup_as:
                func = kws.pop(self._setup_as)
            else:
                func = self._used_functions[-1]
        return func(**kws)