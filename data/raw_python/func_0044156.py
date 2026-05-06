def _execute(self, worker):
        """
        This method is ASSIGNED during the evaluation to control how to resume it once it has been paused
        """
        self._assert_status_is(TaskStatus.RUNNING)
        operation = worker.look_up(self.operation)
        operation.invoke(self, [], worker=worker)