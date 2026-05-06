def help(self, taskname=None):
        """
        List tasks or provide help for specific task
        :param taskname: if supplied, help for this specific task is displayed.
            Otherwise, displays overview of available tasks.
        :return: None
        """

        if not taskname:
            print(inspect.getdoc(self._tasks))
            print()
            print('Available tasks:')
            print()

            for task in self.get_tasks():
                self.print_task_help(task[1], task[0])
        else:
            try:
                task = self.get_task(taskname)
                self.print_task_help(task, task.__name__)

            except TaskNotFound:
                print('Task "%s" not found! Use "ape help" to get usage information.' % taskname)