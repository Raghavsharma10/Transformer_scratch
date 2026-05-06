def _set_tasks_state(self, value):
        """
        Purpose: Set state of all tasks of the current stage.

        :arguments: String
        """
        if value not in states.state_numbers.keys():
            raise ValueError(obj=self._uid,
                             attribute='set_tasks_state',
                             expected_value=states.state_numbers.keys(),
                             actual_value=value)

        for task in self._tasks:
            task.state = value