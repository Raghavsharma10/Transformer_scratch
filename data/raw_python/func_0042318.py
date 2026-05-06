def getCall(self, n): #pylint: disable=invalid-name
        """
        Args:
            n: integer (index of function call)
        Return:
            SpyCall object (or None if the index is not valid)
        """
        call_list = super(SinonSpy, self)._get_wrapper().call_list
        if n >= 0 and n < len(call_list):
            call = call_list[n]
            call.proxy = weakref.proxy(self)
            return call
        else:
            return None