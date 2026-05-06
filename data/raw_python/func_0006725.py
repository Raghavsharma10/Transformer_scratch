def create(
        self,
        name,
        command_to_run,
        container_image,
        container_type,
        description="",
        logs_path="",
        results_path="",
        environment_variables=None,
        required_arguments=None,
        required_arguments_default_values=None,
        extra_data_to_post=None,
    ):
        """Create a container task type.

        Args:
            name (str): The name of the task.
            command_to_run (str): The command to run to execute the task.
            container_image (str): The container name and tag. For
                example, ubuntu:14.04 for Docker; and docker://ubuntu:14:04
                or shub://vsoch/hello-world for Singularity.
            container_type (str): The type of the container.
            description (str, optional): The description of the task type.
            logs_path (str, optional): The path of the logs directory
                inside the container.
            results_path (str, optional): The path of the results
                directory inside the container.
            environment_variables (list, optional): The environment
                variables required on the host to execute the task.
            required_arguments (list, optional): The argument names for
                the task type.
            required_arguments_default_values (dict, optional): Default
                values for the task's required arguments.
            extra_data_to_post (dict, optional): Extra key-value pairs
                to add to the request data. This is useful for
                subclasses which require extra parameters.

        Returns:
            :class:`saltant.models.container_task_type.ContainerTaskType`:
                A container task type model instance representing the
                task type just created.
        """
        # Add in extra data specific to container task types
        if extra_data_to_post is None:
            extra_data_to_post = {}

        extra_data_to_post.update(
            {
                "container_image": container_image,
                "container_type": container_type,
                "logs_path": logs_path,
                "results_path": results_path,
            }
        )

        # Call the parent create function
        return super(ContainerTaskTypeManager, self).create(
            name=name,
            command_to_run=command_to_run,
            description=description,
            environment_variables=environment_variables,
            required_arguments=required_arguments,
            required_arguments_default_values=required_arguments_default_values,
            extra_data_to_post=extra_data_to_post,
        )