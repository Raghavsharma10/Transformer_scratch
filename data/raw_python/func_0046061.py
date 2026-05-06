def set_resource_id(self, resource_id=None):
        """stub"""
        if resource_id is None:
            raise NullArgument()
        if self.get_resource_id_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_id(
                resource_id):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['resourceId'] = resource_id