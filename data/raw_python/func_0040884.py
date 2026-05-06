def from_dict(cls, raw_data, **kwargs):
        """
        This factory for :class:`Model` creates a Model from a dict object.
        """
        instance = cls()
        instance.populate(raw_data, **kwargs)
        instance.validate(**kwargs)
        return instance