def object_result(self):
        """
        Get the object result object, assuming there is only one.  Raises
        an error if there is more than one.
        :return: The result object
        :raises ValueError: If there is more than one result
        """
        num_obj_results = len(self._object_results)

        if num_obj_results < 1:
            return None
        elif num_obj_results < 2:
            return self._object_results[0]
        else:
            raise ValueError("There is more than one result; use 'object_results'")