def wait_until_finished(
        self, uuid, refresh_period=DEFAULT_TASK_INSTANCE_WAIT_REFRESH_PERIOD
    ):
        """Wait until a task instance with the given UUID is finished.

        Args:
            uuid (str): The UUID of the task instance to wait for.
            refresh_period (float, optional): How many seconds to wait
                in between checking the task's status. Defaults to 5
                seconds.

        Returns:
            :class:`saltant.models.base_task_instance.BaseTaskInstance`:
                A task instance model instance representing the task
                instance which we waited for.
        """
        # Wait for the task to finish
        task_instance = self.get(uuid)

        while task_instance.state not in TASK_INSTANCE_FINISH_STATUSES:
            # Wait a bit
            time.sleep(refresh_period)

            # Query again
            task_instance = self.get(uuid)

        return task_instance