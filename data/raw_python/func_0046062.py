def clear_resource_id(self):
        """stub"""
        if (self.get_resource_id_metadata().is_read_only() or
                self.get_resource_id_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['resourceId'] = \
            self.get_resource_id_metadata().get_default_id_values()[0]