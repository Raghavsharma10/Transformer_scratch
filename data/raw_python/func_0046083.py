def clear_decimal_value(self, label):
        """stub"""
        if label not in self.my_osid_object_form._my_map['decimalValues']:
            raise NotFound()
        del self.my_osid_object_form._my_map['decimalValues'][label]