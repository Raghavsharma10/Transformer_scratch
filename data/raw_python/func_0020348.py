def create_role(self, name):
        '''Create a new :class:`Role` owned by this :class:`Subject`'''
        models = self.session.router
        return models.role.new(name=name, owner=self)