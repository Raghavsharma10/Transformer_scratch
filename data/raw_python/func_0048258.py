def wait_and_join(self, task):
        """ Given a task, waits for it until it finishes
        :param task: Task
        :return:
        """
        while not task.has_started:
            time.sleep(self._polling_time)
        task.thread.join()