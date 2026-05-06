def lastCall(self): #pylint: disable=invalid-name
        """
        Return: SpyCall object for this spy's most recent call
        """
        last_index = len(super(SinonSpy, self)._get_wrapper().call_list) - 1
        return self.getCall(last_index)