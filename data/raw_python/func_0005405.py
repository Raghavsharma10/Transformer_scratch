def get_list(self, mutagen_file):
        """Get a list of all values for the field using this style.
        """
        return [self.deserialize(item) for item in self.fetch(mutagen_file)]