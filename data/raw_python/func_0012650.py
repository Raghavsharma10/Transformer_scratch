def xmlrpc_reschedule(self):
        """
        Reschedule all running tasks. 
        """
        if not len(self.scheduled_tasks) == 0:
            self.reschedule = list(self.scheduled_tasks.items())
            self.scheduled_tasks = {}
        return True