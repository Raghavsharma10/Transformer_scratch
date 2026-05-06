def _call_scope(self, scope, *args, **kwargs):
        """
        Call the given model scope.

        :param scope: The scope to call
        :type scope: str
        """
        result = getattr(self._model, scope)(self, *args, **kwargs)

        return result or self