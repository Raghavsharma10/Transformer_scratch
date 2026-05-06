def __get_matching_indices(self, args, kwargs, args_list, kwargs_list):
        """
        Args:
            args: tuple, the arguments inputed by the user
            kwargs: dictionary, the keyword arguments inputed by the user
            args_list: list, a list of argument tuples
            kwargs_list: list, a list of keyword argument dictionaries
        Returns:
            list, the list of indices in args_list/kwargs_list for which the user args/kwargs match
        """
        if args and kwargs:
            if args in args_list and kwargs in kwargs_list:
                args_indices = [i for i, x in enumerate(args_list) if x == args]
                kwargs_indices = [i for i, x in enumerate(kwargs_list) if x == kwargs]
                return list(set(args_indices).intersection(kwargs_indices))
        # args only
        elif args:
            if args in args_list:
                return [i for i, x in enumerate(args_list) if x == args and not kwargs_list[i]]
        #kwargs only
        elif kwargs:
            if kwargs in kwargs_list:
                return [i for i, x in enumerate(kwargs_list) if x == kwargs and not args_list[i]]
        else:
            return []