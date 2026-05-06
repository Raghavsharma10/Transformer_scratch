def run(self):
        """This function needs to be called to start the computation."""
        (task_id, tasks) = self.server.get_task()
        self.task_store.from_dict(tasks)
        for (index, task) in self.task_store:
            result = self.compute(index, task)
            self.results.append(result)
        self.server.task_done((task_id, self.results))