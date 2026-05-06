def get_or_create_application(self, id=None, manifest=None, name=None):
        """ Get application by id or name.
        If not found: create with given or generated parameters
        """
        if id:
            return self.get_application(id=id)
        elif name:
            try:
                app = self.get_application(name=name)
            except exceptions.NotFoundError:
                app = self.create_application(name=name, manifest=manifest)
            return app
        raise exceptions.NotEnoughParams('Not enough parameters')