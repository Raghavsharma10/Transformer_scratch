def _validate_list_type(self, name, obj, *args):
        """
        Helper function that checks the input object type against each in a list of classes, or if the input object
        is a list, each value that it contains against that list.

        :param name: Name of the object.
        :param obj: Object to check the type of.
        :param args: List of classes.
        :raises TypeError: if the input object is not of any of the allowed types.
        """
        if obj is None:
            return
        if isinstance(obj, list):
            for i in obj:
                self._validate_type_not_null(name,  i, *args)
        else:
            self._validate_type(name, obj, *args)