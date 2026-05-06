def selectlanguage(self, event):
        """Store client's selection of a new translation"""

        self.log('Language selection event:', event.client, pretty=True)

        if event.data not in all_languages():
            self.log('Unavailable language selected:', event.data, lvl=warn)
            language = None
        else:
            language = event.data

        if language is None:
            language = 'en'

        event.client.language = language

        if event.client.config is not None:
            event.client.config.language = language
            event.client.config.save()