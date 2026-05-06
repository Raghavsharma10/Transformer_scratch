def is_published(self):
        """stub"""
        if 'published' not in self.my_osid_object._my_map:
            return False
        return bool(self.my_osid_object._my_map['published'])