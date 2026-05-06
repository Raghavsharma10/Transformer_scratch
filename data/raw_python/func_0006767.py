def put(
        self,
        id,
        name,
        description,
        command_to_run,
        environment_variables,
        required_arguments,
        required_arguments_default_values,
        json_file_option,
        extra_data_to_put=None,
    ):
        """Updates a task type on the saltant server.

        Args:
            id (int): The ID of the task type.
            name (str): The name of the task type.
            description (str): The description of the task type.
            command_to_run (str): The command to run to execute the task.
            environment_variables (list): The environment variables
                required on the host to execute the task.
            required_arguments (list): The argument names for the task type.
            required_arguments_default_values (dict): Default values for
                the tasks required arguments.
            json_file_option (str): The name of a command line option,
                e.g., --json-file, which accepts a JSON-encoded file for
                the command to run.
            extra_data_to_put (dict, optional): Extra key-value pairs to
                add to the request data. This is useful for subclasses
                which require extra parameters.
        """
        # Add in extra data specific to container task types
        if extra_data_to_put is None:
            extra_data_to_put = {}

        extra_data_to_put.update({"json_file_option": json_file_option})

        # Call the parent create function
        return super(ExecutableTaskTypeManager, self).put(
            id=id,
            name=name,
            description=description,
            command_to_run=command_to_run,
            environment_variables=environment_variables,
            required_arguments=required_arguments,
            required_arguments_default_values=(
                required_arguments_default_values
            ),
            extra_data_to_put=extra_data_to_put,
        )