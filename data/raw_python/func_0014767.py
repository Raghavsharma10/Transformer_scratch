def _get_run_breadcrumbs(cls, source_type, data_object, task_attempt):
        """Create a path for a given file, in such a way
        that files end up being organized and browsable by run
        """
        # We cannot generate the path unless connect to a TaskAttempt
        # and a run
        if not task_attempt:
            return []
        # If multiple tasks exist, use the original.
        task = task_attempt.tasks.earliest('datetime_created')
        if task is None:
            return []
        run = task.run
        if run is None:
            return []

        breadcrumbs = [
            run.name,
            "task-%s" % str(task.uuid)[0:8],
            "attempt-%s" % str(task_attempt.uuid)[0:8],
        ]

        # Include any ancestors if run is nested
        while run.parent is not None:
            run = run.parent
            breadcrumbs = [run.name] + breadcrumbs

        # Prepend first breadcrumb with datetime and id
        breadcrumbs[0] = "%s-%s-%s" % (
            run.datetime_created.strftime('%Y-%m-%dT%H.%M.%SZ'),
            str(run.uuid)[0:8],
            breadcrumbs[0])

        breadcrumbs = ['runs'] + breadcrumbs
        return breadcrumbs