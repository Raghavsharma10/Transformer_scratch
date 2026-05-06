def put(
        self,
        id,
        name,
        description,
        private,
        runs_executable_tasks,
        runs_docker_container_tasks,
        runs_singularity_container_tasks,
        active,
        whitelists,
    ):
        """Updates a task queue on the saltant server.

        Args:
            id (int): The ID of the task queue.
            name (str): The name of the task queue.
            description (str): The description of the task queue.
            private (bool): A Booleon signalling whether the queue can
                only be used by its associated user.
            runs_executable_tasks (bool): A Boolean specifying whether
                the queue runs executable tasks.
            runs_docker_container_tasks (bool): A Boolean specifying
                whether the queue runs container tasks that run in
                Docker containers.
            runs_singularity_container_tasks (bool): A Boolean
                specifying whether the queue runs container tasks that
                run in Singularity containers.
            active (bool): A Booleon signalling whether the queue is
                active.
            whitelists (list): A list of task whitelist IDs.

        Returns:
            :class:`saltant.models.task_queue.TaskQueue`:
                A task queue model instance representing the task queue
                just updated.
        """
        # Update the object
        request_url = self._client.base_api_url + self.detail_url.format(id=id)
        data_to_put = {
            "name": name,
            "description": description,
            "private": private,
            "runs_executable_tasks": runs_executable_tasks,
            "runs_docker_container_tasks": runs_docker_container_tasks,
            "runs_singularity_container_tasks": runs_singularity_container_tasks,
            "active": active,
            "whitelists": whitelists,
        }

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