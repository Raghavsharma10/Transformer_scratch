def change_task_size(self, size):
        """Blocking request to change number of running tasks"""
        self._pause.value = True
        self.log.debug("About to change task size to {0}".format(size))
        try:
            size = int(size)
        except ValueError:
            self.log.error("Cannot change task size, non integer size provided")
            return False
        if size < 0:
            self.log.error("Cannot change task size, less than 0 size provided")
            return False
        self.max_tasks = size
        if size < self.max_tasks:
            diff = self.max_tasks - size
            self.log.debug("Reducing size offset by {0}".format(diff))
            while True:
                self._update_tasks()
                if len(self.free_tasks) >= diff:
                    for i in range(diff):
                        task_id = self.free_tasks.pop(0)
                        del self.current_tasks[task_id]
                    break
                time.sleep(0.5)
            if not size:
                self._reset_and_pause()
                return True
        elif size > self.max_tasks:
            diff = size - self.max_tasks
            for i in range(diff):
                task_id = str(uuid.uuid4())
                self.current_tasks[task_id] = {}
                self.free_tasks.append(task_id)
        self._pause.value = False
        self.log.debug("Task size changed to {0}".format(size))
        return True