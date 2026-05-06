def styles(self, mutagen_file):
        """Yields the list of storage styles of this field that can
        handle the MediaFile's format.
        """
        for style in self._styles:
            if mutagen_file.__class__.__name__ in style.formats:
                yield style