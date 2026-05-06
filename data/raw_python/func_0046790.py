def get_waypoint_quota_metadata(self):
        """get the metadata for waypoint quota"""
        metadata = dict(self._waypoint_quota_metadata)
        metadata.update({'existing_cardinal_values': self.my_osid_object_form._my_map['waypointQuota']})
        return Metadata(**metadata)