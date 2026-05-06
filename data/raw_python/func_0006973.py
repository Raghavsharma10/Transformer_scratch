def response_data_to_model_instance(self, response_data):
        """Convert response data to a task instance model.

        Args:
            response_data (dict): The data from the request's response.

        Returns:
            :class:`saltant.models.base_task_instance.BaseTaskInstance`:
                A task instance model instance representing the task
                instance from the reponse data.
        """
        # Coerce datetime strings into datetime objects
        response_data["datetime_created"] = dateutil.parser.parse(
            response_data["datetime_created"]
        )

        if response_data["datetime_finished"]:
            response_data["datetime_finished"] = dateutil.parser.parse(
                response_data["datetime_finished"]
            )

        # Instantiate a model for the task instance
        return super(
            BaseTaskInstanceManager, self
        ).response_data_to_model_instance(response_data)