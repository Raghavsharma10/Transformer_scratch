def clear_text(self, label):
        """stub"""
        if label not in self.my_osid_object_form._my_map['texts']:
            raise NotFound()
        del self.my_osid_object_form._my_map['texts'][label]