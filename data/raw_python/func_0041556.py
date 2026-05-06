def act(self, *args, **kwargs):
        """gather a rules parameters together and run the predicate. If that
        returns True, then go on and run the action function

        returns:
            a tuple indicating the results of applying the predicate and the
            action function:
               (False, None) - the predicate failed, action function not run
               (True, True) - the predicate and action functions succeeded
               (True, False) - the predicate succeeded, but the action function
                               failed"""
        pred_args = tuple(args) + tuple(self.predicate_args)
        pred_kwargs = kwargs.copy()
        pred_kwargs.update(self.predicate_kwargs)
        if self.function_invocation_proxy(self.predicate,
                                          pred_args,
                                          pred_kwargs):
            act_args = tuple(args) + tuple(self.action_args)
            act_kwargs = kwargs.copy()
            act_kwargs.update(self.action_kwargs)
            bool_result = self.function_invocation_proxy(self.action, act_args,
                                                         act_kwargs)
            return (True, bool_result)
        else:
            return (False, None)