def languages(self):
        """
        A list of strings describing the user's languages.
        """
        languages = []

        for language in self.cache['languages']:
            language = Structure(
                id = language['id'],
                name = language['name']
            )

            languages.append(language)

        return languages