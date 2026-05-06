def serialize(self, image):
        """Turn a Image into a mutagen.flac.Picture.
        """
        pic = mutagen.flac.Picture()
        pic.data = image.data
        pic.type = image.type_index
        pic.mime = image.mime_type
        pic.desc = image.desc or u''
        return pic