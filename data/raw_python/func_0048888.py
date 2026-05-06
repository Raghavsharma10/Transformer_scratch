def run(self):
        """
        Run the schedule
        """
        self.main_task.thread.start()
        self.main_task.thread.join()