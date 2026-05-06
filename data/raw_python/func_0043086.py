def get_codomain(self, key):
        """
        RETURN AN ARRAY OF OBJECTS THAT key MAPS TO
        """
        return [v for k, v in self.all if k == key]