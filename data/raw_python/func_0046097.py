def has_files(self):
        """stub"""
        # I had to add the following check because file record types
        # don't seem to be implemented
        # correctly for raw edx Question objects
        if 'fileIds' not in self.my_osid_object._my_map:
            return False
        return bool(self.my_osid_object._my_map['fileIds'])