def __get_return_value_no_withargs(self, *args, **kwargs):
        """    
        Pre-conditions:
           (1) The user has created a stub and specified the stub behaviour
           (2) The user has called the stub function with the specified "args" and "kwargs"
           (3) No 'withArgs' conditions were applicable in this case
        Args:
            args: tuple, the arguments inputed by the user
            kwargs: dictionary, the keyword arguments inputed by the user
        Returns:
            any type, the appropriate return value, based on the stub's behaviour setup and the user input
        """
        c = self._conditions
        call_count = self._wrapper.callCount

        # if there might be applicable onCall conditions
        if call_count in c["oncall"]:
            index_list = [i for i, x in enumerate(c["oncall"]) if x and not c["args"][i] and not c["kwargs"][i]]
            for i in reversed(index_list):
                # if the onCall condition applies
                if call_count == c["oncall"][i]:
                    return c["action"][i](*args, **kwargs)

        # else all conditions did not match
        return c["default"](*args, **kwargs)