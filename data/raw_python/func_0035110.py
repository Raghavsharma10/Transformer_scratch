def _iterate_through_class(self, class_dict):
        """Recursive function for output dictionary creation.

        Function will check each value in a dictionary to see if it is a
        class, list, or dictionary object. The idea is to turn all class objects into
        dictionaries. If it is a class object it will pass its ``class.__dict__``
        recursively through this function again. If it is a dictionary,
        it will pass the dictionary recursively through this functin again.

        If the object is a list, it will iterate through entries checking for class
        or dictionary objects and pass them recursively through this function.
        This uses the knowledge of the list structures in the code.

        Args:
            class_dict (obj): Dictionary to iteratively check.

        Returns:
            Dictionary with all class objects turned into dictionaries.

        """
        output_dict = {}
        for key in class_dict:
            val = class_dict[key]
            try:
                val = val.__dict__
            except AttributeError:
                pass

            if type(val) is dict:
                val = self._iterate_through_class(val)

            if type(val) is list:
                temp_val = []
                for val_i in val:
                    try:
                        val_i = val_i.__dict__
                    except AttributeError:
                        pass

                    if type(val_i) is dict:
                        val_i = self._iterate_through_class(val_i)
                    temp_val.append(val_i)
                val = temp_val

            output_dict[key] = val

        return output_dict