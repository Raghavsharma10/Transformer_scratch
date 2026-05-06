def available_languages(self):
        """
        Returns a list of languages providing collection of stop words.
        """
        available_languages = getattr(self, '_available_languages', None)
        if available_languages:
            return available_languages
        try:
            languages = os.listdir(self.data_directory)
        except OSError:
            raise StopWordError(
                "'datas' directory is unreadable, check your installation.")
        languages = sorted(map(lambda x: x.replace('.txt', ''), languages))
        setattr(self, '_available_languages', languages)
        return languages