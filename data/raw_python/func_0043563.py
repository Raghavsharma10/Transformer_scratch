def get_available_name(self, name):
        """
        Deletes the given file if it exists.
        """
        if self.exists(name):
            self.delete(name)
        return name