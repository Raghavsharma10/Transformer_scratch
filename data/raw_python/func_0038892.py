def Execute(self, *params, **kw):
        """Synchronously execute the specified GP task. Parameters are passed
           in either in order or as keywords."""
        fp = self.__expandparamstodict(params, kw)
        return self._get_subfolder('execute/', GPExecutionResult, fp)