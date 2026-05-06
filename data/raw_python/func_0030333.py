def get_task(self, name):
        """Get task by name or create it if it does not exists."""
        if name in self.tasks.keys():
            task = self.tasks[name]
        else:
            task = Task(name)
            self.tasks[name] = task
        return task