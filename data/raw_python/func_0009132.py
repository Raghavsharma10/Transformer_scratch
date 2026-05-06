def raise_exception(self, exception, tup=None):
        """Report an exception back to Storm via logging.

        :param exception: a Python exception.
        :param tup: a :class:`Tuple` object.
        """
        if tup:
            message = (
                "Python {exception_name} raised while processing Tuple "
                "{tup!r}\n{traceback}"
            )
        else:
            message = "Python {exception_name} raised\n{traceback}"
        message = message.format(
            exception_name=exception.__class__.__name__, tup=tup, traceback=format_exc()
        )
        self.send_message({"command": "error", "msg": str(message)})
        self.send_message({"command": "sync"})