def get_tasks(self, list_id, completed=False):
        ''' Gets tasks for the list with the given ID, filtered by the given completion flag '''
        return tasks_endpoint.get_tasks(self, list_id, completed=completed)