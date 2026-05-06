def deserialize(self, mutagen_value):
        """Given a raw value stored on a Mutagen object, decode and
        return the represented value.
        """
        if self.suffix and isinstance(mutagen_value, six.text_type) \
           and mutagen_value.endswith(self.suffix):
            return mutagen_value[:-len(self.suffix)]
        else:
            return mutagen_value