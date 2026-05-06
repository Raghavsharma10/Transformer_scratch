def calledWithExactly(self, *args, **kwargs): #pylint: disable=invalid-name
        """
        Determining whether args/kwargs are fully matched args/kwargs called previously
        Eg.
            f(1, 2, 3)
            spy.alwaysCalledWith(1, 2) will return False, because they are not fully matched
            spy.alwaysCalledWith(1, 2, 3) will return True, because they are fully matched
        Return: Boolean
        """
        self.__remove_args_first_item()
        if args and kwargs:
            return True if (uch.tuple_in_list(self.args, args) and
                            uch.dict_in_list(self.kwargs, kwargs)) else False
        elif args:
            return True if uch.tuple_in_list(self.args, args) else False
        elif kwargs:
            return True if uch.dict_in_list(self.kwargs, kwargs) else False
        else:
            ErrorHandler.called_with_empty_error()