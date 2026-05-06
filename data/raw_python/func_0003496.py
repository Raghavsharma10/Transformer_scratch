def call(self, name, *args, **kwds):
        """
        Call method connected to this handler.

        :type     name: str
        :arg      name: Method name to call.
        :type     args: list
        :arg      args: Arguments for remote method to call.
        :type callback: callable
        :arg  callback: A function to be called with returned value of
                        the remote method.
        :type  errback: callable
        :arg   errback: A function to be called with an error occurred
                        in the remote method.  It is either an instance
                        of :class:`ReturnError` or :class:`EPCError`.

        """
        self.callmanager.call(self, name, *args, **kwds)