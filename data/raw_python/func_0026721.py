def _count_tasks(self):
        """Count the number of tasks, both in the json and directory.

        Returns
        -------
        num_tasks : int
            The total number of all tasks included in the `tasks.json` file.

        """
        self.log.warning("Tasks:")
        tasks, task_names = self.catalog._load_task_list_from_file()
        # Total number of all tasks
        num_tasks = len(tasks)
        # Number which are active by default
        num_tasks_act = len([tt for tt, vv in tasks.items() if vv.active])
        # Number of python files in the tasks directory
        num_task_files = os.path.join(self.catalog.PATHS.tasks_dir, '*.py')
        num_task_files = len(glob(num_task_files))
        tasks_str = "{} ({} default active) with {} task-files.".format(
            num_tasks, num_tasks_act, num_task_files)
        self.log.warning(tasks_str)
        return num_tasks