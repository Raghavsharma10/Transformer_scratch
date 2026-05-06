def update_task_positions_obj(self, positions_obj_id, revision, values):
        '''
        Updates the ordering of tasks in the positions object with the given ID to the ordering in the given values.

        See https://developer.wunderlist.com/documentation/endpoints/positions for more info

        Return:
        The updated TaskPositionsObj-mapped object defining the order of list layout
        '''
        return positions_endpoints.update_task_positions_obj(self, positions_obj_id, revision, values)