def _validate_nested_list_type(self, name, obj, nested_level, *args):
        """
        Helper function that checks the input object as a list then recursively until nested_level is 1.

        :param name: Name of the object.
        :param obj: Object to check the type of.
        :param nested_level: Integer with the current nested level.
        :param args: List of classes.
        :raises TypeError: if the input object is not of any of the allowed types.
        """
        if nested_level <= 1:
            self._validate_list_type(name, obj, *args)
        else:
            if obj is None:
                return
            if not isinstance(obj, list):
                raise TypeError(self.__class__.__name__ + '.' + name + ' contains value of type ' +
                                type(obj).__name__ + ' where a list is expected')
            for sub_obj in obj:
                self._validate_nested_list_type(name, sub_obj, nested_level - 1, *args)