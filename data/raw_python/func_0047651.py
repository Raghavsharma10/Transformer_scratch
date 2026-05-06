def has_preview(self):
        """stub"""
        # I had to add the following check because file record types don't seem to be implemented
        # correctly for raw edx Question objects
        if ('fileIds' not in self.my_osid_object._my_map or
                'preview' not in self.my_osid_object._my_map['fileIds'] or
                self.my_osid_object._my_map['fileIds']['preview'] is None):
            return False
        return bool(self.my_osid_object._my_map['fileIds']['preview'])