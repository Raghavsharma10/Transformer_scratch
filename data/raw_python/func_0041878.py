def update_short(self, **kwargs):
        """
        Update the short optional arguments (those with one leading '-')

        This method updates the short argument name for the specified function
        arguments as stored in :attr:`unfinished_arguments`

        Parameters
        ----------
        ``**kwargs``
            Keywords must be keys in the :attr:`unfinished_arguments`
            dictionary (i.e. keywords of the root functions), values the short
            argument names

        Examples
        --------
        Setting::

            >>> parser.update_short(something='s', something_else='se')

        is basically the same as::

            >>> parser.update_arg('something', short='s')
            >>> parser.update_arg('something_else', short='se')

        which in turn is basically comparable to::

            >>> parser.add_argument('-s', '--something', ...)
            >>> parser.add_argument('-se', '--something_else', ...)

        See Also
        --------
        update_shortf, update_long"""
        for key, val in six.iteritems(kwargs):
            self.update_arg(key, short=val)