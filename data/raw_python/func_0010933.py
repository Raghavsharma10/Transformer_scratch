def connect(self, signal, slot, transform=None, condition=None):
        """Defines a connection between this objects signal and another objects slot

           signal: the signal this class will emit, to cause the slot method to be called
           receiver: the object containing the slot method to be called
           slot: the slot method to call
           transform: an optional value override to pass into the slot method as the first variable
           condition: only call the slot if the value emitted matches the required value or calling required returns True
        """
        if not signal in self.signals:
            print("WARNING: {0} is trying to connect a slot to an undefined signal: {1}".format(self.__class__.__name__,
                                                                                       str(signal)))
            return

        if not hasattr(self, 'connections'):
            self.connections = {}
        connection = self.connections.setdefault(signal, {})
        connection = connection.setdefault(condition, {})
        connection[slot] = transform