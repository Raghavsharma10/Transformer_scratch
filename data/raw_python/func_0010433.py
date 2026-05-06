def configure(self, **_options):
        '''
        Sets language-specific configuration.

        Raises TTSError on error.
        '''
        language, voice, voiceinfo, options = self._configure(**_options)
        self.languages_options[language] = (voice, options)