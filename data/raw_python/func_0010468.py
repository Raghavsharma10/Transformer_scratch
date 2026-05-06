def _apply_task(task: Task, args: Tuple, kwargs: Dict[str, Any]) -> Any:
        """Logs the time spent while running the task."""
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        start = monotonic()
        try:
            return task.apply(*args, **kwargs)
        finally:
            delta = monotonic() - start
            logger.info("%s finished in %i seconds." % (task.name, delta))