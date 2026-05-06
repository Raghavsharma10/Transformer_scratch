def get_decimal_value(self, label):
        """stub"""
        if self.has_decimal_value(label):
            return float(self.my_osid_object._my_map['decimalValues'][label])
        raise IllegalState()