def put(self):
        """Updates this task queue on the saltant server.

        Returns:
            :class:`saltant.models.task_queue.TaskQueue`:
                A task queue model instance representing the task queue
                just updated.
        """
        return self.manager.put(
            id=self.id,
            name=self.name,
            description=self.description,
            private=self.private,
            runs_executable_tasks=self.runs_executable_tasks,
            runs_docker_container_tasks=self.runs_docker_container_tasks,
            runs_singularity_container_tasks=self.runs_singularity_container_tasks,
            active=self.active,
            whitelists=self.whitelists,
        )