def wait_until_finished(
        self, refresh_period=DEFAULT_TASK_INSTANCE_WAIT_REFRESH_PERIOD
    ):
        """Wait until a task instance with the given UUID is finished.

        Args:
            refresh_period (int, optional): How many seconds to wait
                before checking the task's status. Defaults to 5
                seconds.

        Returns:
            :class:`saltant.models.base_task_instance.BaseTaskInstance`:
                This task instance model after it finished.
        """
        return self.manager.wait_until_finished(
            uuid=self.uuid, refresh_period=refresh_period
        )