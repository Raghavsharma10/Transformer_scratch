def unregisterAPI(self, name):
        """
        Remove an API from this handler
        """
        if name.startswith('public/'):
            target = 'public'
            name = name[len('public/'):]
        else:
            target = self.servicename
            name = name
        removes = [m for m in self.handler.handlers.keys() if m.target == target and m.name == name]
        for m in removes:
            self.handler.unregisterHandler(m)