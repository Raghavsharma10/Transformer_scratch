def update_subtask(self, subtask_id, revision, title=None, completed=None):
        '''
        Updates the subtask with the given ID

        See https://developer.wunderlist.com/documentation/endpoints/subtask for detailed parameter information

        Returns:
        Subtask with given ID with properties and revision updated
        '''
        return subtasks_endpoint.update_subtask(self, subtask_id, revision, title=title, completed=completed)