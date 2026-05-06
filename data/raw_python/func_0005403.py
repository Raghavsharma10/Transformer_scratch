def set(self, mutagen_file, value):
        """Assign the value for the field using this style.
        """
        self.store(mutagen_file, self.serialize(value))