def register(cls, instance_class, name=None):
        """Register a class with the factory.

        :param instance_class: the class to register with the factory (not a
            string)
        :param name: the name to use as the key for instance class lookups;
            defaults to the name of the class

        """
        if name is None:
            name = instance_class.__name__
        cls.INSTANCE_CLASSES[name] = instance_class