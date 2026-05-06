def _get_future_tasks(self):
        """Assemble a list of future alerts"""

        self.alerts = {}
        now = std_now()

        for task in objectmodels['task'].find({'alert_time': {'$gt': now}}):
            self.alerts[task.alert_time] = task

        self.log('Found', len(self.alerts), 'future tasks')