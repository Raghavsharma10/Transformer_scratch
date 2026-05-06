def reload(self):
        """ Function reload
        Sync the full object
        """
        self.load(self.api.get(self.objName, self.key))