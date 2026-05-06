def set_attributes(self, **kwargs):
        """
        Set the resource attributes from the kwargs.
        Only sets items in the `self.Meta.attributes` white list.

        Args:
            kwargs: Keyword arguements passed into the init of this class
        """
        for field, value in kwargs.items():
            if field in self.Meta.attributes:
                setattr(self, field, value)