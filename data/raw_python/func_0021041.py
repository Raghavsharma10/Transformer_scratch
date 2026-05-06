def intersection_update(self, *others):
        """
        Update the set, keeping only elements found in it and all *others*.

        :param others: Iterables, each one as a single positional argument.
        :rtype: None

        .. note::
            The same behavior as at :func:`difference_update` applies.
        """
        return self._op_update_helper(
            tuple(others), operator.and_, 'sinterstore', update=True
        )