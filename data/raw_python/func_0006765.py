def put(self):
        """Updates this task type on the saltant server.

        Returns:
            :class:`saltant.models.container_task_type.ExecutableTaskType`:
                An executable task type model instance representing the task type
                just updated.
        """
        return self.manager.put(
            id=self.id,
            name=self.name,
            description=self.description,
            command_to_run=self.command_to_run,
            environment_variables=self.environment_variables,
            required_arguments=self.required_arguments,
            required_arguments_default_values=(
                self.required_arguments_default_values
            ),
            json_file_option=self.json_file_option,
        )