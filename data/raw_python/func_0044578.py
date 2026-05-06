def _get_stop_words(self, language):
        """
        Internal method for getting the stop words collections
        and raising errors.
        """
        if language not in self.available_languages:
            raise StopWordError(
                'Stop words are not available in "%s".\n'
                'If possible do a pull request at : '
                'https://github.com/Fantomas42/mots-vides' %
                language)
        try:
            filename = self.get_collection_filename(language)
            collection = self.read_collection(filename)
        except IOError:
            raise StopWordError(
                '"%s" file is unreadable, check your installation.' %
                filename)
        return collection