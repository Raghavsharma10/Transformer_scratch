def clear_integer_value(self, label):
        """stub"""
        if label not in self.my_osid_object_form._my_map['integerValues']:
            raise NotFound()
        del self.my_osid_object_form._my_map['integerValues'][label]