def get_integer_value(self, label):
        """stub"""
        if self.has_integer_value(label):
            return int(self.my_osid_object._my_map['integerValues'][label])
        raise IllegalState()