def set_hook_after_request(self, func):
        """Run the specified function/symbol (C level) after each request.

        :param str|unicode func:

        """
        self._set('after-request-hook', func, multi=True)

        return self._section