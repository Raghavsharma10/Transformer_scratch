def execute_callback(self, *args, **kwargs):
        """Executes a callback and returns the proper response.

        Refer to :meth:`sijax.Sijax.execute_callback` for more details.
        """
        response = self._sijax.execute_callback(*args, **kwargs)
        return _make_response(response)