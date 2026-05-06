def filter(self, filter_arguments):
        """
        Takes a dictionary of filter parameters.
        Return a list of objects based on a list of parameters.
        """
        results = self._get_content()

        # Filter based on a dictionary of search parameters
        if isinstance(filter_arguments, dict):
            for item, content in iteritems(self._get_content()):
                for key, value in iteritems(filter_arguments):
                    keys = key.split('.')
                    value = filter_arguments[key]

                    if not self._contains_value({item: content}, keys, value):
                        del results[item]

        # Filter based on an input string that should match database key
        if isinstance(filter_arguments, str):
            if filter_arguments in results:
                return [{filter_arguments: results[filter_arguments]}]
            else:
                return []

        return results