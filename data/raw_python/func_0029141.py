def _validate_type(self, name, obj, *args):
        """
        Helper function that checks the input object type against each in a list of classes. This function
        also allows the input value to be equal to None.

        :param name: Name of the object.
        :param obj: Object to check the type of.
        :param args: List of classes.
        :raises TypeError: if the input object is not of any of the allowed types.
        """
        if obj is None:
            return
        for arg in args:
            if isinstance(obj, arg):
                return
        raise TypeError(self.__class__.__name__ + '.' + name + ' is of type ' + type(obj).__name__ +
                        '. Must be equal to None or one of the following types: ' + str(args))