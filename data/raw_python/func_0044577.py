def get_stop_words(self, language, fail_safe=False):
        """
        Returns a StopWord object initialized with the stop words collection
        requested by ``language``.
        If the requested language is not available a StopWordError is raised.
        If ``fail_safe`` is set to True, an empty StopWord object is returned.
        """
        try:
            language = self.language_codes[language]
        except KeyError:
            pass

        collection = self.LOADED_LANGUAGES_CACHE.get(language)

        if collection is None:
            try:
                collection = self._get_stop_words(language)
                self.LOADED_LANGUAGES_CACHE[language] = collection
            except StopWordError as error:
                if not fail_safe:
                    raise error
                collection = []

        stop_words = StopWord(language, collection)
        return stop_words