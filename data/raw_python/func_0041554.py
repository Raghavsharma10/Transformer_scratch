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
        if self.predicate(*args, **kwargs):
            bool_result = self.action(*args, **kwargs)
            return (True, bool_result)
        else:
            return (False, None)