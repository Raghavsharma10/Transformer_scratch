def add_zone(self, spatial_unit, container_id, name='', description='', visible=True, reuse=0, drop_behavior_type=None):
        """container_id is a targetId that the zone belongs to
        """
        if not isinstance(spatial_unit, abc_mapping_primitives.SpatialUnit):
            raise InvalidArgument('zone is not a SpatialUnit')
        # if not isinstance(name, DisplayText):
        #     raise InvalidArgument('name is not a DisplayText object')
        if not isinstance(reuse, int):
            raise InvalidArgument('reuse must be an integer')
        if reuse < 0:
            raise InvalidArgument('reuse must be >= 0')
        if not isinstance(name, DisplayText):
            # if default ''
            name = self._str_display_text(name)
        if not isinstance(description, DisplayText):
            # if default ''
            description = self._str_display_text(description)
        zone = {
            'id': str(ObjectId()),
            'spatialUnit': spatial_unit.get_spatial_unit_map(),
            'containerId': container_id,
            'names': [self._dict_display_text(name)],
            'descriptions': [self._dict_display_text(description)],
            'visible': visible,
            'reuse': reuse,
            'dropBehaviorType': str(drop_behavior_type)
        }
        self.my_osid_object_form._my_map['zones'].append(zone)
        return zone