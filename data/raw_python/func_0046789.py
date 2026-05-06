def set_max_waypoint_items(self, max_waypoint_items):
        """This determines how many waypoint items will be seen for a scaffolded wrong answer"""
        if self.get_max_waypoint_items_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_cardinal(max_waypoint_items,
                                                           self.get_max_waypoint_items_metadata()):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['maxWaypointItems'] = max_waypoint_items