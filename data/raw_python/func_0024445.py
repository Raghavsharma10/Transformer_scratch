def _update_task(self, task):
        """
        Assigns current task step to self.task
        then updates the task's data with self.task_data

        Args:
            task: Task object.
        """
        self.task = task
        self.task.data.update(self.task_data)
        self.task_type = task.task_spec.__class__.__name__
        self.spec = task.task_spec
        self.task_name = task.get_name()
        self.activity = getattr(self.spec, 'service_class', '')
        self._set_lane_data()