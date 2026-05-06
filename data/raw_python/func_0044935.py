def to_serializable_value(self):
        """
        Run through all fields of the object and parse the values

        :return:
        :rtype: dict
        """
        return {
            name: field.to_serializable_value()
            for name, field in self.value.__dict__.items()
            if isinstance(field, Field) and self.value
        }