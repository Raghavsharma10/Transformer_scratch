def _is_match(self, response, answer):
        """Does the response match the answer """

        def compare_conditions(droppable_id, spatial_units, response_conditions):
            """Compare response coordinates with spatial units for droppable_id"""
            coordinate_match = True
            for coordinate in response_conditions['coordinate_conditions']['include'][droppable_id]:
                answer_match = False
                for spatial_unit in spatial_units:
                    if (coordinate['containerId'] == spatial_unit['containerId'] and
                            coordinate['coordinate'] in spatial_unit['spatialUnit']):
                        answer_match = True
                        break
                coordinate_match = coordinate_match and answer_match
            return coordinate_match

        # Did the consumer application already do the work for us?
        if response.has_zone_conditions():
            return bool(response.get_zone_conditions() == answer.get_zone_conditions())

        answer_conditions = self._get_conditions_map(answer)
        response_conditions = self._get_conditions_map(response)

        # Check to see if the lists of droppables used are the same:
        if set(answer_conditions['spatial_unit_conditions']['include']) != set(response_conditions['coordinate_conditions']['include']):
            return False

        # Compare included answer spatial unit areas to response coordinates
        for droppable_id, spatial_units in answer_conditions['spatial_unit_conditions']['include'].items():
            # Do the number of defined include conditions match:
            if len(spatial_units) != len(response_conditions['coordinate_conditions']['include'][droppable_id]):
                return False
            if not compare_conditions(droppable_id, spatial_units, response_conditions):
                return False

        # Compare excluded answer spatial unit areas to response coordinates
        for droppable_id, spatial_units in answer_conditions['spatial_unit_conditions']['exclude'].items():
            if compare_conditions(droppable_id, spatial_units, response_conditions):
                return False
        return True