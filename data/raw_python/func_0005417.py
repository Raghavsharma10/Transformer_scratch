def save(self):
        """Write the object's tags back to the file. May
        throw `UnreadableFileError`.
        """
        # Possibly save the tags to ID3v2.3.
        kwargs = {}
        if self.id3v23:
            id3 = self.mgfile
            if hasattr(id3, 'tags'):
                # In case this is an MP3 object, not an ID3 object.
                id3 = id3.tags
            id3.update_to_v23()
            kwargs['v2_version'] = 3

        mutagen_call('save', self.path, self.mgfile.save, **kwargs)