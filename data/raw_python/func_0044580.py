def get_collection_filename(self, language):
        """
        Returns the filename containing the stop words collection
        for a specific language.
        """
        filename = os.path.join(self.data_directory, '%s.txt' % language)
        return filename