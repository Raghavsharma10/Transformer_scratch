def call_sync(self, name, args, timeout=None):
        """
        Blocking version of :meth:`call`.

        :type    name: str
        :arg     name: Remote function name to call.
        :type    args: list
        :arg     args: Arguments passed to the remote function.
        :type timeout: int or None
        :arg  timeout: Timeout in second.  None means no timeout.

        If the called remote function raise an exception, this method
        raise an exception.  If you give `timeout`, this method may
        raise an `Empty` exception.

        """
        return self._blocking_request(self.call, timeout, name, args)