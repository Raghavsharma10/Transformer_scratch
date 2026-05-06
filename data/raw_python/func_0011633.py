def get_list_subtasks(self, list_id, completed=False):
        ''' Gets subtasks for the list with given ID '''
        return subtasks_endpoint.get_list_subtasks(self, list_id, completed=completed)