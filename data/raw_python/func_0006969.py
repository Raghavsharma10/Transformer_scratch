def create(self, task_type_id, task_queue_id, arguments=None, name=""):
        """Create a task instance.

        Args:
            task_type_id (int): The ID of the task type to base the task
                instance on.
            task_queue_id (int): The ID of the task queue to run the job
                on.
            arguments (dict, optional): The arguments to give the task
                type.
            name (str, optional): A non-unique name to give the task
                instance.

        Returns:
            :class:`saltant.models.base_task_instance.BaseTaskInstance`:
                A task instance model instance representing the task
                instance just created.
        """
        # Make arguments an empty dictionary if None
        if arguments is None:
            arguments = {}

        # Create the object
        request_url = self._client.base_api_url + self.list_url
        data_to_post = {
            "name": name,
            "arguments": json.dumps(arguments),
            "task_type": task_type_id,
            "task_queue": task_queue_id,
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