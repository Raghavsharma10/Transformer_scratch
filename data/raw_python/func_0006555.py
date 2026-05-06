def create(
        self,
        name,
        command_to_run,
        description="",
        environment_variables=None,
        required_arguments=None,
        required_arguments_default_values=None,
        extra_data_to_post=None,
    ):
        """Create a task type.

        Args:
            name (str): The name of the task.
            command_to_run (str): The command to run to execute the task.
            description (str, optional): The description of the task type.
            environment_variables (list, optional): The environment
                variables required on the host to execute the task.
            required_arguments (list, optional): The argument names for
                the task type.
            required_arguments_default_values (dict, optional): Default
                values for the tasks required arguments.
            extra_data_to_post (dict, optional): Extra key-value pairs
                to add to the request data. This is useful for
                subclasses which require extra parameters.

        Returns:
            :class:`saltant.models.base_task_instance.BaseTaskType`:
                A task type model instance representing the task type
                just created.
        """
        # Set None for optional list and dicts to proper datatypes
        if environment_variables is None:
            environment_variables = []

        if required_arguments is None:
            required_arguments = []

        if required_arguments_default_values is None:
            required_arguments_default_values = {}

        # Create the object
        request_url = self._client.base_api_url + self.list_url
        data_to_post = {
            "name": name,
            "description": description,
            "command_to_run": command_to_run,
            "environment_variables": json.dumps(environment_variables),
            "required_arguments": json.dumps(required_arguments),
            "required_arguments_default_values": json.dumps(
                required_arguments_default_values
            ),
        }

        # Add in extra data if any was passed in
        if extra_data_to_post is not None:
            data_to_post.update(extra_data_to_post)

        response = self._client.session.post(request_url, data=data_to_post)

        # Validate that the request was successful
        self.validate_request_success(
            response_text=response.text,
            request_url=request_url,
            status_code=response.status_code,
            expected_status_code=HTTP_201_CREATED,
        )

        # Return a model instance representing the task type
        return self.response_data_to_model_instance(response.json())