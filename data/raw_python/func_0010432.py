def configure_default(self, **_options):
        '''
        Sets default configuration.

        Raises TTSError on error.
        '''
        language, voice, voiceinfo, options = self._configure(**_options)
        self.languages_options[language] = (voice, options)
        self.default_language = language
        self.default_options = options