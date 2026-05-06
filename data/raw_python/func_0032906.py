def get_listeners(self, name):
        """
        Return the callables related to name
        """        
        return list(map(lambda listener: listener[0], self.listeners[name]))