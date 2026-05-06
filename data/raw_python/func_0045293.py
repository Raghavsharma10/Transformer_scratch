def to_serializable_value(self):
        """
        Parses the Hightonmodel to a serializable value such dicts, lists, strings
        This can be used to save the model in a NoSQL database

        :return: the serialized HightonModel
        :rtype: dict
        """
        return_dict = {}
        for name, field in self.__dict__.items():
            if isinstance(field, fields.Field):
                return_dict[name] = field.to_serializable_value()
        return return_dict