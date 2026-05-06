def get_spatial_unit_conditions(self):
        """stub"""
        condition_list = deepcopy(self.my_osid_object._my_map['spatialUnitConditions'])
        for condition in condition_list:
            condition['spatialUnit'] = SpatialUnitFactory().get_spatial_unit(spatial_unit_map=condition['spatialUnit'])
        return condition_list