def wait(self, method, *args):
        """
        Call a method on the zombie.js Browser instance and wait on a callback.

        :param method: the method to call, e.g., html()
        :param args: one of more arguments for the method
        """
        methodargs = encode_args(args, extra=True)
        js = """
        %s(%s wait_callback);
        """ % (method, methodargs)
        self._send(js)