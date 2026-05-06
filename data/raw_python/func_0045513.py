def run_tasks(self):
        '''
        Runs all assigned task in separate green threads. If the task should not be run, schedule it
        '''
        pool = Pool(len(self.tasks))
        for task in self.tasks:
            # Launch a green thread to schedule the task
            # A task will be managed by 2 green thread: execution thread and scheduling thread
            pool.spawn(self.run, task)
        return pool