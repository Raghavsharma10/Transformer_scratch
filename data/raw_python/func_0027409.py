def _remap(object, name, value, safe=True):
    """Prevent accidental assignment of existing members

    Arguments:
        object (object): Parent of new attribute
        name (str): Name of new attribute
        value (object): Value of new attribute
        safe (bool): Whether or not to guarantee that
            the new attribute was not overwritten.
            Can be set to False under condition that
            it is superseded by extensive testing.

    """

    if os.getenv("QT_TESTING") is not None and safe:
        # Cannot alter original binding.
        if hasattr(object, name):
            raise AttributeError("Cannot override existing name: "
                                 "%s.%s" % (object.__name__, name))

        # Cannot alter classes of functions
        if type(object).__name__ != "module":
            raise AttributeError("%s != 'module': Cannot alter "
                                 "anything but modules" % object)

    elif hasattr(object, name):
        # Keep track of modifications
        self.__modified__.append(name)

    self.__remapped__.append(name)

    setattr(object, name, value)