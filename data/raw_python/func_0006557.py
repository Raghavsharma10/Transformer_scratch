def response_data_to_model_instance(self, response_data):
        """Convert response data to a task type model.

        Args:
            response_data (dict): The data from the request's response.

        Returns:
            :class:`saltant.models.base_task_type.BaseTaskType`:
                A model instance representing the task type from the
                reponse data.
        """
        # Coerce datetime strings into datetime objects
        response_data["datetime_created"] = dateutil.parser.parse(
            response_data["datetime_created"]
        )

        # Instantiate a model for the task instance
        return super(
            BaseTaskTypeManager, self
        ).response_data_to_model_instance(response_data)