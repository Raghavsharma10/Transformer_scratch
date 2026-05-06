def create(
        self,
        name,
        command_to_run,
        description="",
        environment_variables=None,
        required_arguments=None,
        required_arguments_default_values=None,
        json_file_option=None,
        extra_data_to_post=None,
    ):
        """Create a container task type.

        Args:
            name (str): The name of the task.
            command_to_run (str): The command to run to execute the task.
            description (str, optional): The description of the task type.
            environment_variables (list, optional): The environment
                variables required on the host to execute the task.
            required_arguments (list, optional): The argument names for
                the task type.
            required_arguments_default_values (dict, optional): Default
                values for the task's required arguments.
            json_file_option (str, optional): The name of a command line
                option, e.g., --json-file, which accepts a JSON-encoded
                file for the command to run.
            extra_data_to_post (dict, optional): Extra key-value pairs
                to add to the request data. This is useful for
                subclasses which require extra parameters.

        Returns:
            :class:`saltant.models.container_task_type.ExecutableTaskType`:
                An executable task type model instance representing the
                task type just created.
        """
        # Add in extra data specific to container task types
        if extra_data_to_post is None:
            extra_data_to_post = {}

        extra_data_to_post.update({"json_file_option": json_file_option})

        # Call the parent create function
        return super(ExecutableTaskTypeManager, self).create(
            name=name,
            command_to_run=command_to_run,
            description=description,
            environment_variables=environment_variables,
            required_arguments=required_arguments,
            required_arguments_default_values=required_arguments_default_values,
            extra_data_to_post=extra_data_to_post,
        )