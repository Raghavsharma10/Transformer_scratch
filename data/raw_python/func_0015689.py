def connect_after(self, detailed_signal, handler, *args):
        """connect_after(detailed_signal: str, handler: function, *args) -> handler_id: int

        The connect_after() method is similar to the connect() method
        except that the handler is added to the signal handler list after
        the default class signal handler. Otherwise the details of handler
        definition and invocation are the same.
        """

        flags = GConnectFlags.CONNECT_AFTER
        return self.__connect(flags, detailed_signal, handler, *args)