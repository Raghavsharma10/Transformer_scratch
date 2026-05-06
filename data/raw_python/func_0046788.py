def get_max_waypoint_items_metadata(self):
        """get the metadata for max waypoint items"""
        metadata = dict(self._max_waypoint_items_metadata)
        metadata.update({'existing_cardinal_values': self.my_osid_object_form._my_map['maxWaypointItems']})
        return Metadata(**metadata)