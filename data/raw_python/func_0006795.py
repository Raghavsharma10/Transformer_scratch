def patch(
        self,
        id,
        name=None,
        description=None,
        whitelisted_container_task_types=None,
        whitelisted_executable_task_types=None,
    ):
        """Partially updates a task whitelist on the saltant server.

        Args:
            id (int): The ID of the task whitelist.
            name (str, optional): The name of the task whitelist.
            description (str, optional): A description of the task whitelist.
            whitelisted_container_task_types (list, optional): A list of
                whitelisted container task type IDs.
            whitelisted_executable_task_types (list, optional): A list
                of whitelisted executable task type IDs.

        Returns:
            :class:`saltant.models.task_whitelist.TaskWhitelist`:
                A task whitelist model instance representing the task
                whitelist just updated.
        """
        # Update the object
        request_url = self._client.base_api_url + self.detail_url.format(id=id)

        data_to_patch = {}

        if name is not None:
            data_to_patch["name"] = name

        if description is not None:
            data_to_patch["description"] = description

        if whitelisted_container_task_types is not None:
            data_to_patch[
                "whitelisted_container_task_types"
            ] = whitelisted_container_task_types

        if whitelisted_executable_task_types is not None:
            data_to_patch[
                "whitelisted_executable_task_types"
            ] = whitelisted_executable_task_types

        response = self._client.session.patch(request_url, data=data_to_patch)

        # Validate that the request was successful
        self.validate_request_success(
            response_text=response.text,
            request_url=request_url,
            status_code=response.status_code,
            expected_status_code=HTTP_200_OK,
        )

        # Return a model instance representing the task instance
        return self.response_data_to_model_instance(response.json())