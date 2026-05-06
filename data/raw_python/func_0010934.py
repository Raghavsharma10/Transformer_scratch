def disconnect(self, signal=None, slot=None, transform=None, condition=None):
        """Removes connection(s) between this objects signal and connected slot(s)

           signal: the signal this class will emit, to cause the slot method to be called
           receiver: the object containing the slot method to be called
           slot: the slot method or function to call
           transform: an optional value override to pass into the slot method as the first variable
           condition: only call the slot method if the value emitted matches this condition
        """
        if slot:
            self.connections[signal][condition].pop(slot, None)
        elif condition is not None:
            self.connections[signal].pop(condition, None)
        elif signal:
            self.connections.pop(signal, None)
        else:
            delattr(self, 'connections')