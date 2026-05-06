def serialize(self, image):
        """Return an APIC frame populated with data from ``image``.
        """
        assert isinstance(image, Image)
        frame = mutagen.id3.Frames[self.key]()
        frame.data = image.data
        frame.mime = image.mime_type
        frame.desc = image.desc or u''

        # For compatibility with OS X/iTunes prefer latin-1 if possible.
        # See issue #899
        try:
            frame.desc.encode("latin-1")
        except UnicodeEncodeError:
            frame.encoding = mutagen.id3.Encoding.UTF16
        else:
            frame.encoding = mutagen.id3.Encoding.LATIN1

        frame.type = image.type_index
        return frame