def get_id(self):
        """override get_id to generate our "magic" id that encodes scaffolding information"""
        waypoint_index = 0
        if 'waypointIndex' in self.my_osid_object._my_map:
            waypoint_index = self.my_osid_object._my_map['waypointIndex']
        # NOTE that the order of the dict **must** match the order in generate_children()
        #   when creating the child_part_id
        #   1) level
        #   2) objective_ids
        #   3) parent_id
        #   4) waypoint_index
        magic_identifier = OrderedDict({
            'level': self._level,
            'objective_ids': self.my_osid_object._my_map['learningObjectiveIds'],
        })
        if self._magic_parent_id is not None:
            magic_identifier['parent_id'] = str(self._magic_parent_id)
        magic_identifier['waypoint_index'] = waypoint_index

        identifier = quote('{0}?{1}'.format(str(self.my_osid_object._my_map['_id']),
                                            json.dumps(magic_identifier)))
        return Id(namespace='assessment_authoring.AssessmentPart',
                  identifier=identifier,
                  authority=MAGIC_PART_AUTHORITY)