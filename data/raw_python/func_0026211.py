def set_stream(self, stream):
        """
        Set the stream that this logger is meant to replace.  Usually this will
        be either `sys.stdout` or `sys.stderr`, but can be any object with
        `write()` and `flush()` methods, as supported by
        `logging.StreamHandler`.
        """

        for handler in self.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                self.handlers.remove(handler)

        if stream is not None:
            stream_handler = logging.StreamHandler(stream)
            stream_handler.addFilter(_StreamHandlerEchoFilter())
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            self.addHandler(stream_handler)

        self.stream = stream