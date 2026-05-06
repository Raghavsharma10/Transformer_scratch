def update(self, *others):
        """
        Update the set, adding elements from all *others*.

        :param others: Iterables, each one as a single positional argument.
        :rtype: None

        .. note::
            If all *others* are :class:`Set` instances, the operation
            is performed completely in Redis. Otherwise, values are retrieved
            from Redis and the operation is performed in Python.
        """
        return self._op_update_helper(
            tuple(others), operator.or_, 'sunionstore', update=True
        )