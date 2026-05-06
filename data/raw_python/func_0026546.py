def unregister(self):
        """Removes the unique name from the systems unique name list"""
        self.names.remove(self.uniquename)
        super(ConfigurableMeta, self).unregister()