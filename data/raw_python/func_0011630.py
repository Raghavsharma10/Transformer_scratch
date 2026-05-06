def update_task(self, task_id, revision, title=None, assignee_id=None, completed=None, recurrence_type=None, recurrence_count=None, due_date=None, starred=None, remove=None):
        ''' 
        Updates the task with the given ID to have the given information 
        
        NOTE: The 'remove' parameter is an optional list of parameters to remove from the given task, e.g. ['due_date']
        '''
        return tasks_endpoint.update_task(self, task_id, revision, title=title, assignee_id=assignee_id, completed=completed, recurrence_type=recurrence_type, recurrence_count=recurrence_count, due_date=due_date, starred=starred, remove=remove)