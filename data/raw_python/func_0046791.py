def set_waypoint_quota(self, waypoint_quota):
        """how many waypoint questions need to be answered correctly"""
        if self.get_waypoint_quota_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_cardinal(waypoint_quota,
                                                           self.get_waypoint_quota_metadata()):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['waypointQuota'] = waypoint_quota