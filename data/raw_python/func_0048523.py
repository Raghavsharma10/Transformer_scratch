def add_spatial_unit_condition(self, droppable_id, container_id, spatial_unit, match=True):
        """stub"""
        if not isinstance(spatial_unit, abc_mapping_primitives.SpatialUnit):
            raise InvalidArgument('spatial_unit is not a SpatialUnit')

        self.my_osid_object_form._my_map['spatialUnitConditions'].append(
            {'droppableId': droppable_id, 'containerId': container_id, 'spatialUnit': spatial_unit.get_spatial_unit_map(), 'match': match})
        self.my_osid_object_form._my_map['spatialUnitConditions'].sort()