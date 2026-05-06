def _run(self):
        """ Run the task respecting dependencies
        """
        for node in self.node.relatives:
            launch_node_task(node)
        for node in self.node.relatives:
            self.wait_and_join(node.task)
        if self.node.parent:
            while not self.node.parent.task.siblings_permission:
                time.sleep(self._polling_time)
        self.has_started = True
        self.main()
        self.siblings_permission = True
        for node in self.node.siblings:
            launch_node_task(node)
        for node in self.node.siblings:
            self.wait_and_join(node.task)
        self.finished_at = time.time()
        self.scheduler.notify_execution(self)
        self.has_finished = True