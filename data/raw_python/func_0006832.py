def create(
        self,
        name,
        description="",
        private=False,
        runs_executable_tasks=True,
        runs_docker_container_tasks=True,
        runs_singularity_container_tasks=True,
        active=True,
        whitelists=None,
    ):
        """Create a task queue.

        Args:
            name (str): The name of the task queue.
            description (str, optional): A description of the task queue.
            private (bool, optional): A boolean specifying whether the
                queue is exclusive to its creator. Defaults to False.
            runs_executable_tasks (bool, optional): A Boolean specifying
                whether the queue runs executable tasks. Defaults to
                True.
            runs_docker_container_tasks (bool, optional): A Boolean
                specifying whether the queue runs container tasks that
                run in Docker containers. Defaults to True.
            runs_singularity_container_tasks (bool, optional): A Boolean
                specifying whether the queue runs container tasks that
                run in Singularity containers. Defaults to True.
            active (bool, optional): A boolean specifying whether the
                queue is active. Default to True.
            whitelists (list, optional): A list of task whitelist IDs.
                Defaults to None (which gets translated to []).

        Returns:
            :class:`saltant.models.task_queue.TaskQueue`:
                A task queue model instance representing the task queue
                just created.
        """
        # Translate whitelists None to [] if necessary
        if whitelists is None:
            whitelists = []

        # Create the object
        request_url = self._client.base_api_url + self.list_url
        data_to_post = {
            "name": name,
            "description": description,
            "private": private,
            "runs_executable_tasks": runs_executable_tasks,
            "runs_docker_container_tasks": runs_docker_container_tasks,
            "runs_singularity_container_tasks": runs_singularity_container_tasks,
            "active": active,
            "whitelists": whitelists,
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