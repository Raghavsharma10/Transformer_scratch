def get_default_sample(self):
        """Return default value for the element
        """
        if self.type not in Object.Types or self.type is Object.Types.type:
            return self.type_object.get_sample()
        else:
            return self.get_object().get_sample()