def get(self, id):
        """Get the model instance with a given id.

        Args:
            id (int or str): The primary identifier (e.g., pk or UUID)
                for the task instance to get.

        Returns:
            :class:`saltant.models.resource.Model`:
                A :class:`saltant.models.resource.Model` subclass
                instance representing the resource requested.
        """
        # Get the object
        request_url = self._client.base_api_url + self.detail_url.format(id=id)

        response = self._client.session.get(request_url)

        # Validate that the request was successful
        self.validate_request_success(
            response_text=response.text,
            request_url=request_url,
            status_code=response.status_code,
            expected_status_code=HTTP_200_OK,
        )

        # Return a model instance
        return self.response_data_to_model_instance(response.json())