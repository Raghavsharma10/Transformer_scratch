def get(self, id=None, name=None):
        """Get a task queue.

        Either the id xor the name of the task type must be specified.

        Args:
            id (int, optional): The id of the task type to get.
            name (str, optional): The name of the task type to get.

        Returns:
            :class:`saltant.models.task_queue.TaskQueue`:
                A task queue model instance representing the task queue
                requested.

        Raises:
            ValueError: Neither id nor name were set *or* both id and
                name were set.
        """
        # Validate arguments - use an xor
        if not (id is None) ^ (name is None):
            raise ValueError("Either id or name must be set (but not both!)")

        # If it's just ID provided, call the parent function
        if id is not None:
            return super(TaskQueueManager, self).get(id=id)

        # Try getting the task queue by name
        return self.list(filters={"name": name})[0]