def reload(self):
        """ Function reload
        Reload the full object to ensure sync
        """
        realData = self.load()
        self.clear()
        self.update(realData)