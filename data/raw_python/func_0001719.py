def register(self, name, fun, description=None):
        """ Register function on this service """
        self.methods[name] = fun
        self.descriptions[name] = description