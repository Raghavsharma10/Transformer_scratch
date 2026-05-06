def parse_known_chained(self, args=None):
        """
        Parse the argument directly to the function used for setup

        This function parses the command line arguments to the function that
        has been used for the :meth:`setup_args` method.


        Parameters
        ----------
        args: list
            The arguments parsed to the :meth:`parse_args` function

        Returns
        -------
        argparse.Namespace
            The namespace with mapping from command name to the function
            return
        list
            The remaining arguments that could not be interpreted

        See also
        --------
        parse_known
        """
        ns, remainder = self.parse_known_args(args)
        kws = vars(ns)
        return self._parse2subparser_funcs(kws), remainder