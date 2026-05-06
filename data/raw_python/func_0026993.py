def is_previous_task_processing(self, *args, **kwargs):
        """ Return True if exist task that is equal to current and is uncompleted """
        app = self._get_app()
        inspect = app.control.inspect()
        active = inspect.active() or {}
        scheduled = inspect.scheduled() or {}
        reserved = inspect.reserved() or {}
        uncompleted = sum(list(active.values()) + list(scheduled.values()) + reserved.values(), [])
        return any(self.is_equal(task, *args, **kwargs) for task in uncompleted)