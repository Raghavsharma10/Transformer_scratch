def add_listener(self, name, listener, priority=0):
        """
        Add a new listener to the dispatch
        """
        if name not in self.listeners:
            self.listeners[name] = []

        self.listeners[name].append((listener, priority))

        # reorder event
        self.listeners[name].sort(key=lambda listener: listener[1], reverse=True)