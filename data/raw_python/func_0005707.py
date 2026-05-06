def add_new_repr_method(cls):
    """
    Add default __repr__ method in case no user-defined one is present.
    """

    if isinstance(cls.__repr__, WrapperDescriptorType):
        cls.__repr__ = lambda self: f"<{self.__class__.__name__}, id={hex(id(self))}>"
    else:
        # Keep the user-defined __repr__ method
        pass