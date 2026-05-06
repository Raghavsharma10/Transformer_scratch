def calledWithMatch(self, *args, **kwargs): #pylint: disable=invalid-name
        """
        Determining whether args/kwargs are matched args/kwargs called previously
        Handle each arg/kwarg as a SinonMatcher
        Eg.
            f(1, 2, 3)
            spy.alwaysCalledWith(1, 2, 3) will return True, because they are fully matched
            spy.alwaysCalledWith(int) will return True, because type are partially matched
        Return: Boolean

        Note: sinon.js have no definition of combination case, here is current implementation:

            for args or kwargs, it should be matched in each individual call
            Eg. func(1,2,3) -> func(4,5,6)
                spy.calledWithMatch(1,5) is not valid
            Eg. func(a=1,b=2,c=3) -> func(a=4,b=5,c=6)
                spy.calledWithMatch(a=1,b=5,c=6) is not valid

            however, for combination case, it should be matched separated
            Eg. func(1,2,c=3) -> func(2,b=5,c=6)
                spy.calledWithMatch(1,2,c=3) is valid,
                    because spy.calledWithMatch(1,2) and spy.calledWithMatch(c=3) are valid
                spy.calledWithMatch(1,c=6) is valid,
                    because spy.calledWithMatch(1) and spy.calledWithMatch(c=6) are valid
                spy.calledWithMatch(1,2,c=6) is valid,
                    because spy.calledWithMatch(1,2) and spy.calledWithMatch(c=6) are valid
        """
        self.__remove_args_first_item()
        if args and kwargs:
            return (uch.tuple_partial_cmp(args, self.args, self.__get_func) and
                    uch.dict_partial_cmp(kwargs, self.kwargs, self.__get_func))
        elif args:
            return uch.tuple_partial_cmp(args, self.args, self.__get_func)
        elif kwargs:
            return uch.dict_partial_cmp(kwargs, self.kwargs, self.__get_func)
        else:
            ErrorHandler.called_with_empty_error()
        self.__get_func = SinonSpy.__get_by_matcher