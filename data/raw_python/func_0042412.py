def __get_call_count(self, args, kwargs, args_list, kwargs_list):
        """
        Args:
            args: tuple, the arguments inputed by the user
            kwargs: dictionary, the keyword arguments inputed by the user
            args_list: list, the tuples of args from all the times this stub was called
            kwargs_list: list, the dictionaries of kwargs from all the times this stub was called
        Returns:
            integer, the number of times this combination of args/kwargs has been called
        """
        return len(self.__get_matching_indices(args, kwargs, args_list, kwargs_list))