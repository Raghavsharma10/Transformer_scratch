def parse_chained(self, args=None):
        """
        Parse the argument directly to the function used for setup

        This function parses the command line arguments to the function that
        has been used for the :meth:`setup_args`.


        Parameters
        ----------
        args: list
            The arguments parsed to the :meth:`parse_args` function

        Returns
        -------
        argparse.Namespace
            The namespace with mapping from command name to the function
            return

        See also
        --------
        parse_known_chained
        """
        kws = vars(self.parse_args(args))
        return self._parse2subparser_funcs(kws)