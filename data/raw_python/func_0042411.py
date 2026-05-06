def __get_matching_withargs_indices(self, *args, **kwargs):
        """
        Args:
            args: tuple, the arguments inputed by the user
            kwargs: dictionary, the keyword arguments inputed by the user
        Returns:
            list, the list of indices in conditions for which the user args/kwargs match
        """
        return self.__get_matching_indices(args, kwargs, self._conditions["args"], self._conditions["kwargs"])