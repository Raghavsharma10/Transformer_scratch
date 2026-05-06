def prune(self):
        """
        Remove any tasks that have stubs as ancestors (and the stubs
        themselves).

        Returns the set of nodes which were removed.
        """
        pruned = set()
        stubs = frozenset(self._stubs)

        for stub in stubs:
            pruned.update(self.remove(stub, strategy=Strategy.remove))

        return pruned - stubs