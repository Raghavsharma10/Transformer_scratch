def set_list(self, mutagen_file, values):
        """Set all values for the field using this style. `values`
        should be an iterable.
        """
        self.store(mutagen_file, [self.serialize(value) for value in values])