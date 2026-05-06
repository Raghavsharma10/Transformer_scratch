def patch(
        self,
        id,
        name=None,
        description=None,
        private=None,
        runs_executable_tasks=None,
        runs_docker_container_tasks=None,
        runs_singularity_container_tasks=None,
        active=None,
        whitelists=None,
    ):
        """Partially updates a task queue on the saltant server.

        Args:
            id (int): The ID of the task queue.
            name (str, optional): The name of the task queue.
            description (str, optional): The description of the task
                queue.
            private (bool, optional): A Booleon signalling whether the
                queue can only be used by its associated user.
            runs_executable_tasks (bool, optional): A Boolean specifying
                whether the queue runs executable tasks.
            runs_docker_container_tasks (bool, optional): A Boolean
                specifying whether the queue runs container tasks that
                run in Docker containers.
            runs_singularity_container_tasks (bool, optional): A Boolean
                specifying whether the queue runs container tasks that
                run in Singularity containers.
            active (bool, optional): A Booleon signalling whether the
                queue is active.
            whitelists (list, optional): A list of task whitelist IDs.

        Returns:
            :class:`saltant.models.task_queue.TaskQueue`:
                A task queue model instance representing the task queue
                just updated.
        """
        # Update the object
        request_url = self._client.base_api_url + self.detail_url.format(id=id)

        data_to_patch = {}

        if name is not None:
            data_to_patch["name"] = name

        if description is not None:
            data_to_patch["description"] = description

        if private is not None:
            data_to_patch["private"] = private

        if runs_executable_tasks is not None:
            data_to_patch["runs_executable_tasks"] = runs_executable_tasks

        if runs_docker_container_tasks is not None:
            data_to_patch[
                "runs_docker_container_tasks"
            ] = runs_docker_container_tasks

        if runs_singularity_container_tasks is not None:
            data_to_patch[
                "runs_singularity_container_tasks"
            ] = runs_singularity_container_tasks

        if active is not None:
            data_to_patch["active"] = active

        if whitelists is not None:
            data_to_patch["whitelists"] = whitelists

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