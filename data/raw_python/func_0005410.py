def store(self, mutagen_file, pictures):
        """``pictures`` is a list of mutagen.flac.Picture instances.
        """
        mutagen_file.clear_pictures()
        for pic in pictures:
            mutagen_file.add_picture(pic)