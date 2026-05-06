def import_from_file(self, index, filename):
        """Import this instrument's settings from the given file. Will
        automatically add the instrument's synth and table to the song's
        synths and tables if needed.

        Note that this may invalidate existing instrument accessor objects.

        :param index: the index into which to import

        :param filename: the file from which to load

        :raises ImportException: if importing failed, usually because the song
          doesn't have enough synth or table slots left for the instrument's
          synth or table
        """

        with open(filename, 'r') as fp:
            self._import_from_struct(index, json.load(fp))