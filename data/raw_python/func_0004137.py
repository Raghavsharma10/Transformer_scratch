def create_dictionary_of_element_from_dictionary(self, property_name, datas):
        """Populate a dictionary of elements
        """
        response = {}
        if property_name in datas and datas[property_name] is not None and isinstance(datas[property_name], collections.Iterable):
            for key, value in datas[property_name].items():
                response[key] = self.create_from_name_and_dictionary(key, value)

        return response