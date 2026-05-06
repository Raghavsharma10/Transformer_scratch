def difference_update(self, *others):
        """
        Update the set, removing elements found in *others*.

        :param others: Iterables, each one as a single positional argument.
        :rtype: None

        .. note::
            The same behavior as at :func:`update` applies.
        """
        return self._op_update_helper(
            tuple(others), operator.sub, 'sdiffstore', update=True
        )