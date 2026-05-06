def xmlrpc_status(self):
        """
        Return a status message
        """
        return ("""
        %i Jobs are still wating for execution
        %i Jobs are being processed
        %i Jobs are done
        """ %(self.task_store.partitions - 
                self.results - 
                len(self.scheduled_tasks),
              len(self.scheduled_tasks),
              self.results))