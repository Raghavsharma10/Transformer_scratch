def get_coordinate_conditions(self):
        """stub"""
        condition_list = deepcopy(self.my_osid_object._my_map['coordinateConditions'])
        for condition in condition_list:
            condition['coordinate'] = BasicCoordinate(condition['coordinate'])
        return condition_list