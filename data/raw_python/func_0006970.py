def clone(self, uuid):
        """Clone the task instance with given UUID.

        Args:
            uuid (str): The UUID of the task instance to clone.

        Returns:
            :class:`saltant.models.base_task_instance.BaseTaskInstance`:
                A task instance model instance representing the task
                instance created due to the clone.
        """
        # Clone the object
        request_url = self._client.base_api_url + self.clone_url.format(
            id=uuid
        )

        response = self._client.session.post(request_url)

        # Validate that the request was successful
        self.validate_request_success(
            response_text=response.text,
            request_url=request_url,
            status_code=response.status_code,
            expected_status_code=HTTP_201_CREATED,
        )

        # Return a model instance
        return self.response_data_to_model_instance(response.json())