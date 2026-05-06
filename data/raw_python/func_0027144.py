def _compute_args(self, data=dict(), **kwargs):
        """ Compute the arguments

            Try to import attributes from data.
            Otherwise compute kwargs arguments.

            Args:
                data: a dict()
                kwargs: a list of arguments
        """

        for name, remote_attribute in self._attributes.items():
            default_value = BambouConfig.get_default_attribute_value(self.__class__, name, remote_attribute.attribute_type)
            setattr(self, name, default_value)

        if len(data) > 0:
            self.from_dict(data)

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)