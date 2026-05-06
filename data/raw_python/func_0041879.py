def update_long(self, **kwargs):
        """
        Update the long optional arguments (those with two leading '-')

        This method updates the short argument name for the specified function
        arguments as stored in :attr:`unfinished_arguments`

        Parameters
        ----------
        ``**kwargs``
            Keywords must be keys in the :attr:`unfinished_arguments`
            dictionary (i.e. keywords of the root functions), values the long
            argument names

        Examples
        --------
        Setting::

            >>> parser.update_long(something='s', something_else='se')

        is basically the same as::

            >>> parser.update_arg('something', long='s')
            >>> parser.update_arg('something_else', long='se')

        which in turn is basically comparable to::

            >>> parser.add_argument('--s', dest='something', ...)
            >>> parser.add_argument('--se', dest='something_else', ...)

        See Also
        --------
        update_short, update_longf"""
        for key, val in six.iteritems(kwargs):
            self.update_arg(key, long=val)