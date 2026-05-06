def put(
        self,
        id,
        name,
        description,
        command_to_run,
        environment_variables,
        required_arguments,
        required_arguments_default_values,
        logs_path,
        results_path,
        container_image,
        container_type,
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
            extra_data_to_put (dict, optional): Extra key-value pairs to
                add to the request data. This is useful for subclasses
                which require extra parameters.
            logs_path (str): The path of the logs directory inside the
                container.
            results_path (str): The path of the results directory inside
                the container.
            container_image (str): The container name and tag. For
                example, ubuntu:14.04 for Docker; and docker://ubuntu:14:04
                or shub://vsoch/hello-world for Singularity.
            container_type (str): The type of the container.
        """
        # Add in extra data specific to container task types
        if extra_data_to_put is None:
            extra_data_to_put = {}

        extra_data_to_put.update(
            {
                "logs_path": logs_path,
                "results_path": results_path,
                "container_image": container_image,
                "container_type": container_type,
            }
        )

        # Call the parent create function
        return super(ContainerTaskTypeManager, self).put(
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