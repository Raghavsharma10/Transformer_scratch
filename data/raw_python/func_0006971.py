def terminate(self, uuid):
        """Terminate the task instance with given UUID.

        Args:
            uuid (str): The UUID of the task instance to terminate.

        Returns:
            :class:`saltant.models.base_task_instance.BaseTaskInstance`:
                A task instance model instance representing the task
                instance that was told to terminate.
        """
        # Clone the object
        request_url = self._client.base_api_url + self.terminate_url.format(
            id=uuid
        )

        response = self._client.session.post(request_url)

        # Validate that the request was successful
        self.validate_request_success(
            response_text=response.text,
            request_url=request_url,
            status_code=response.status_code,
            expected_status_code=HTTP_202_ACCEPTED,
        )

        # Return a model instance
        return self.response_data_to_model_instance(response.json())