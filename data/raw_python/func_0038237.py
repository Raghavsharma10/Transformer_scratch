def get_attribute_keys(self):
        """
        Returns a list of managed attributes for the Model class

        Implemented for use with data adapters, can be used to quickly make a list of the
        attribute names in a prestans model
        """

        _attribute_keys = list()

        for attribute_name, type_instance in self.getmembers():

            if isinstance(type_instance, DataType):
                _attribute_keys.append(attribute_name)

        return _attribute_keys