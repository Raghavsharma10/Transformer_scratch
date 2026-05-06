def filter_tasks(self, task_names, keep_dependencies=False):
        """If filter is applied only tasks with given name and its dependencies (if keep_keep_dependencies=True) are kept in the list of tasks."""
        new_tasks = {}
        for task_name in task_names:
            task = self.get_task(task_name)
            if task not in new_tasks:
                new_tasks[task.name] = task
            if keep_dependencies:
                for dependency in task.ordered_dependencies():
                    if dependency not in new_tasks:
                        new_tasks[dependency.name] = dependency
            else:
                #strip dependencies
                task.dependencies = set()

        self.tasks = new_tasks