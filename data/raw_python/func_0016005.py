def object_results(self, object_results):
        """
        Set the results to an iterable of values.  The values will be collected
        into a list.  A single value is allowed; it will be converted to a
        length 1 list.
        :param object_results: The results to set
        """
        if _is_iterable_non_string(object_results):
            self._object_results = list(object_results)
        elif object_results is None:
            self._object_results = []
        else:
            self._object_results = [object_results]