def iter_search_nodes(self, **conditions):
        """
        Search nodes in an interative way. Matches are being yield as
        they are being found. This avoids to scan the full tree
        topology before returning the first matches. Useful when
        dealing with huge trees.
        """
        for n in self.traverse():
            conditions_passed = 0
            for key, value in six.iteritems(conditions):
                if hasattr(n, key) and getattr(n, key) == value:
                    conditions_passed +=1
            if conditions_passed == len(conditions):
                yield n