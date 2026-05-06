def xmlrpc_task_done(self, result):
        """
        Take the results of a computation and put it into the results list.
        """
        (task_id, task_results) = result
        del self.scheduled_tasks[task_id]
        self.task_store.update_results(task_id, task_results)
        self.results += 1
        return True