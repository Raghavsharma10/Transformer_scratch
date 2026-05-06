def get_attributes(input_dict):
        """
        Get attributes of xml tags in input_dict and creates a dictionary with the attribute name as the key and the
        attribute value as the value
        :param input_dict: The xml tag with the attributes and values
        :return: dict
        """
        return {k.lstrip("@"): v for k, v in input_dict.items() if k[0] == "@"}