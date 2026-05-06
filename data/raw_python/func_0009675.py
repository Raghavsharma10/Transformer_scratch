def is_ready(self, check=True):
        """Return ``True`` if the results are ready.

        If you pass ``check=False``, no attempt is made to check again for
        results.

        :param bool check: whether to query for the results
        :return: ``True`` if the results are ready
        :rtype: bool
        """
        if not self._is_ready and check:
            self.check_if_ready()
        return self._is_ready