def call(self, jsondata):
        """
        Calls jsonrpc service's method and returns its return value in a JSON
        string or None if there is none.

        Arguments:
        jsondata -- remote method call in jsonrpc format
        """
        result = yield self.call_py(jsondata)
        if result is None:
            defer.returnValue(None)
        else:
            defer.returnValue(json.dumps(result))