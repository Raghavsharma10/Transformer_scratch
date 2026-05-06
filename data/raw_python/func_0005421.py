def add_field(cls, name, descriptor):
        """Add a field to store custom tags.

        :param name: the name of the property the field is accessed
                     through. It must not already exist on this class.

        :param descriptor: an instance of :class:`MediaField`.
        """
        if not isinstance(descriptor, MediaField):
            raise ValueError(
                u'{0} must be an instance of MediaField'.format(descriptor))
        if name in cls.__dict__:
            raise ValueError(
                u'property "{0}" already exists on MediaField'.format(name))
        setattr(cls, name, descriptor)