def serialize(self, image):
        """Turn a Image into a base64 encoded FLAC picture block.
        """
        pic = mutagen.flac.Picture()
        pic.data = image.data
        pic.type = image.type_index
        pic.mime = image.mime_type
        pic.desc = image.desc or u''

        # Encoding with base64 returns bytes on both Python 2 and 3.
        # Mutagen requires the data to be a Unicode string, so we decode
        # it before passing it along.
        return base64.b64encode(pic.write()).decode('ascii')