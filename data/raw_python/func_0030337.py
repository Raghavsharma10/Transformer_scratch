def tsort(self):
        """Given a partial ordering, return a totally ordered list.

        part is a dict of partial orderings.  Each value is a set,
        which the key depends on.

        The return value is a list of sets, each of which has only
        dependencies on items in previous entries in the list.

        raise ValueError if ordering is not possible (check for circular or missing dependencies)"""

        task_dict = {}
        for key, task in self.tasks.iteritems():
            task_dict[task] = task.dependencies
        # parts = parts.copy()
        parts = task_dict.copy()

        result = []
        while True:
            level = set([name for name, deps in parts.iteritems() if not deps])
            if not level:
                break
            result.append(level)
            parts = dict([(name, deps - level) for name, deps in parts.iteritems() if name not in level])
        if parts:
            raise ValueError('total ordering not possible (check for circular or missing dependencies)')
        return result