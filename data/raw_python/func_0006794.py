def create(
        self,
        name,
        description="",
        whitelisted_container_task_types=None,
        whitelisted_executable_task_types=None,
    ):
        """Create a task whitelist.

        Args:
            name (str): The name of the task whitelist.
            description (str, optional): A description of the task whitelist.
            whitelisted_container_task_types (list, optional): A list of
                whitelisted container task type IDs.
            whitelisted_executable_task_types (list, optional): A list
                of whitelisted executable task type IDs.

        Returns:
            :class:`saltant.models.task_whitelist.TaskWhitelist`:
                A task whitelist model instance representing the task
                whitelist just created.
        """
        # Translate whitelists None to [] if necessary
        if whitelisted_container_task_types is None:
            whitelisted_container_task_types = []

        if whitelisted_executable_task_types is None:
            whitelisted_executable_task_types = []

        # Create the object
        request_url = self._client.base_api_url + self.list_url
        data_to_post = {
            "name": name,
            "description": description,
            "whitelisted_container_task_types": whitelisted_container_task_types,
            "whitelisted_executable_task_types": whitelisted_executable_task_types,
        }

        response = self._client.session.post(request_url, data=data_to_post)

        # Validate that the request was successful
        self.validate_request_success(
            response_text=response.text,
            request_url=request_url,
            status_code=response.status_code,
            expected_status_code=HTTP_201_CREATED,
        )

        # Return a model instance representing the task instance
        return self.response_data_to_model_instance(response.json())