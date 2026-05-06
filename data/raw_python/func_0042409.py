def __default_custom_function(self, *args, **kwargs):
        """
        If the user does not specify a custom function with which to replace the original,
        then this is the function that we shall use. This function allows the user to call
        returns/throws to customize the return value.
        
        Args:
            args: tuple, the arguments inputed by the user
            kwargs: dictionary, the keyword arguments inputed by the user
        Returns:
            anything, the return values specified by the conditions
                      (i.e. what the user defined with returns/throws)
        """
        index_list = self.__get_matching_withargs_indices(*args, **kwargs)
        # if there are 'withArgs' conditions that might be applicable
        if index_list:
            return self.__get_return_value_withargs(index_list, *args, **kwargs)
        # else no 'withArgs' conditions are applicable
        else:
            return self.__get_return_value_no_withargs(*args, **kwargs)