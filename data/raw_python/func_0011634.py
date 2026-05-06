def create_subtask(self, task_id, title, completed=False):
        ''' 
        Creates a subtask with the given title under the task with the given ID 
        
        Return:
        Newly-created subtask
        '''
        return subtasks_endpoint.create_subtask(self, task_id, title, completed=completed)