def put(self):
        """Updates this task whitelist on the saltant server.

        Returns:
            :class:`saltant.models.task_whitelist.TaskWhitelist`:
                A task whitelist model instance representing the task
                whitelist just updated.
        """
        return self.manager.put(
            id=self.id,
            name=self.name,
            description=self.description,
            whitelisted_container_task_types=(
                self.whitelisted_container_task_types
            ),
            whitelisted_executable_task_types=(
                self.whitelisted_executable_task_types
            ),
        )