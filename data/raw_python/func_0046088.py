def add_weight(self, weight):
        """stub"""
        if weight is None:
            raise NullArgument('weight cannot be None')
        if not self.my_osid_object_form._is_valid_decimal(
                weight, self.get_weight_metadata()):
            raise InvalidArgument('weight')
        self.my_osid_object_form._my_map['weight'] = weight