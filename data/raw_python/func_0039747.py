def get_tasks(self):
        """
        Return tasks as list of (name, function) tuples.
        """

        def predicate(item):
            return (inspect.isfunction(item) and
                    item.__name__ not in self._helper_names)
        return inspect.getmembers(self._tasks, predicate)