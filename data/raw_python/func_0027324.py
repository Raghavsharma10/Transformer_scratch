def save_srm(self, filename):
        """Save a project in .srm format to the target file.

        :param filename: the name of the file to which to save
        """
        with open(filename, 'wb') as fp:
            raw_data = bread.write(self._song_data, spec.song)
            fp.write(raw_data)