def create_list_of_element_from_dictionary(self, property_name, datas):
        """Populate a list of elements
        """
        response = []
        if property_name in datas and datas[property_name] is not None and isinstance(datas[property_name], list):
            for value in datas[property_name]:
                response.append(self.create_from_dictionary(value))

        return response