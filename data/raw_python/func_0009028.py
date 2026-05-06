def _done_callback(self, wrapped):
        """Internal "done callback" to set the result of the object.

        The result of the object if forced by the wrapped future. So this
        internal callback must be called when the wrapped future is ready.

        Args:
            wrapped (Future): the wrapped Future object
        """
        if wrapped.exception():
            self.set_exception(wrapped.exception())
        else:
            self.set_result(wrapped.result())