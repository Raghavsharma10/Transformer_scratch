def assert_chain_calls(self, *calls):
        """
        Asserts that a chained method was called (parents in the chain do not
        matter, nor are they tracked).  Use with `mock.call`.

        >>> obj.filter(foo='bar').select_related('baz')
        >>> obj.assert_chain_calls(mock.call.filter(foo='bar'))
        >>> obj.assert_chain_calls(mock.call.select_related('baz'))
        >>> obj.assert_chain_calls(mock.call.reverse())
        *** AssertionError: [call.reverse()] not all found in call list, ...

        """

        all_calls = self.__parent.mock_calls[:]

        not_found = []
        for kall in calls:
            try:
                all_calls.remove(kall)
            except ValueError:
                not_found.append(kall)
        if not_found:
            if self.__parent.mock_calls:
                message = '%r not all found in call list, %d other(s) were:\n%r' % (not_found, len(self.__parent.mock_calls), self.__parent.mock_calls)
            else:
                message = 'no calls were found'

            raise AssertionError(message)