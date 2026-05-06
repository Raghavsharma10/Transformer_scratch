def _pass_uid(self):
        """
        Purpose: Assign the parent Stage and the parent Pipeline to all the tasks of the current stage. 

        :arguments: set of Tasks (optional)
        :return: list of updated Tasks
        """

        for task in self._tasks:
            task.parent_stage['uid'] = self._uid
            task.parent_stage['name'] = self._name
            task.parent_pipeline['uid'] = self._p_pipeline['uid']
            task.parent_pipeline['name'] = self._p_pipeline['name']