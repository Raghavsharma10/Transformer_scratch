def get_task_signature(cls, instance, serialized_instance, **kwargs):
        """
        Delete each resource using specific executor.
        Convert executors to task and combine all deletion task into single sequential task.
        """
        cleanup_tasks = [
            ProjectResourceCleanupTask().si(
                core_utils.serialize_class(executor_cls),
                core_utils.serialize_class(model_cls),
                serialized_instance,
            )
            for (model_cls, executor_cls) in cls.executors
        ]

        if not cleanup_tasks:
            return core_tasks.EmptyTask()

        return chain(cleanup_tasks)