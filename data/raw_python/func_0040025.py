def set_refresh(self, timeout, callback, *callback_args):
        """
        It is just stub for simplify setting timeout.
        Args:
          timeout (int): timeout in milliseconds, after which callback will be called
          callback (callable): usually, just a function that will be called each time after timeout
          *callback_args (any type): arguments that will be passed to callback function
        """
        GObject.timeout_add(timeout, callback, *callback_args)