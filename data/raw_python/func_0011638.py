def update_subtask_positions_obj(self, positions_obj_id, revision, values):
        '''
        Updates the ordering of subtasks in the positions object with the given ID to the ordering in the given values.

        See https://developer.wunderlist.com/documentation/endpoints/positions for more info

        Return:
        The updated SubtaskPositionsObj-mapped object defining the order of list layout
        '''
        return positions_endpoints.update_subtask_positions_obj(self, positions_obj_id, revision, values)