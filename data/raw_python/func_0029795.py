def wait_for_tasks(self, raise_if_error=True):
        """
        Wait for the running tasks lauched from the sessions.

        Note that it also wait for tasks that are started from other tasks
        callbacks, like on_finished.

        :param raise_if_error: if True, raise all possible encountered
            errors using :class:`TaskErrors`. Else the errors are returned
            as a list.
        """
        errors = []
        tasks_seen = TaskCache()
        while True:
            for session in self.values():
                errs = session.wait_for_tasks(raise_if_error=False)
                errors.extend(errs)
            # look for tasks created after the wait (in callbacks of
            # tasks from different sessions)
            tasks = []
            for session in self.values():
                tasks.extend(session.tasks())
            # if none, then just break - else loop to wait for them
            if not any(t for t in tasks if t not in tasks_seen):
                break
        if raise_if_error and errors:
            raise TaskErrors(errors)
        return errors