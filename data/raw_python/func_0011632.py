def get_task_subtasks(self, task_id, completed=False):
        ''' Gets subtasks for task with given ID '''
        return subtasks_endpoint.get_task_subtasks(self, task_id, completed=completed)