def __get_return_value_withargs(self, index_list, *args, **kwargs):
        """    
        Pre-conditions:
           (1) The user has created a stub and specified the stub behaviour
           (2) The user has called the stub function with the specified "args" and "kwargs"
           (3) One or more 'withArgs' conditions were applicable in this case
        Args:
            index_list: list, the list of indices in conditions for which the user args/kwargs match
            args: tuple, the arguments inputed by the user
            kwargs: dictionary, the keyword arguments inputed by the user
        Returns:
            any type, the appropriate return value, based on the stub's behaviour setup and the user input
        """
        c = self._conditions
        args_list = self._wrapper.args_list
        kwargs_list = self._wrapper.kwargs_list

        # indices with an arg and oncall have higher priority and should be checked first
        indices_with_oncall = [i for i in reversed(index_list) if c["oncall"][i]]

        # if there are any combined withArgs+onCall conditions
        if indices_with_oncall:
            call_count = self.__get_call_count(args, kwargs, args_list, kwargs_list)
            for i in indices_with_oncall:
                if c["oncall"][i] == call_count:
                    return c["action"][i](*args, **kwargs)

        # else if there are simple withArgs conditions
        indices_without_oncall = [i for i in reversed(index_list) if not c["oncall"][i]]
        if indices_without_oncall:
            max_index = max(indices_without_oncall)
            return c["action"][max_index](*args, **kwargs)

        # else all conditions did not match
        return c["default"](*args, **kwargs)