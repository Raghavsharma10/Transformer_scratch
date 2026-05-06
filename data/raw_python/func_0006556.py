def put(
        self,
        id,
        name,
        description,
        command_to_run,
        environment_variables,
        required_arguments,
        required_arguments_default_values,
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

        Returns:
            :class:`saltant.models.base_task_type.BaseTaskType`:
                A :class:`saltant.models.base_task_type.BaseTaskType`
                subclass instance representing the task type just
                updated.
        """
        # Update the object
        request_url = self._client.base_api_url + self.detail_url.format(id=id)
        data_to_put = {
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
        if extra_data_to_put is not None:
            data_to_put.update(extra_data_to_put)

        response = self._client.session.put(request_url, data=data_to_put)

        # Validate that the request was successful
        self.validate_request_success(
            response_text=response.text,
            request_url=request_url,
            status_code=response.status_code,
            expected_status_code=HTTP_200_OK,
        )

        # Return a model instance representing the task instance
        return self.response_data_to_model_instance(response.json())