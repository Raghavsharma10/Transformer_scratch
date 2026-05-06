def get_next(self):
        """Return next task from the stack that has all dependencies resolved.
        Return None if there are no tasks with resolved dependencies or is there are no more tasks on stack.
        Use `count` to check is there are still some task left on the stack.

        raise ValueError if total ordering is not possible."""

        self.update_tasks_status()

        if self.dirty:
            self.tsort()
            self.dirty = False

        for key, task in self.tasks.iteritems():
            if task.is_new() and task.has_resolved_dependencies():
                return task

        return None