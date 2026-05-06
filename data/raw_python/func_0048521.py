def add_coordinate_condition(self, droppable_id, container_id, coordinate, match=True):
        """stub"""
        if not isinstance(coordinate, BasicCoordinate):
            raise InvalidArgument('coordinate is not a BasicCoordinate')
        self.my_osid_object_form._my_map['coordinateConditions'].append(
            {'droppableId': droppable_id, 'containerId': container_id, 'coordinate': coordinate.get_values(), 'match': match})
        self.my_osid_object_form._my_map['coordinateConditions'].sort(key=lambda k: k['containerId'])