def say(self, phrase, **_options):
        '''
        Says the phrase, optionally allows to select/override any voice options.
        '''
        language, voice, voiceinfo, options = self._configure(**_options)
        self._logger.debug("Saying '%s' with '%s'", phrase, self.SLUG)
        self._say(phrase, language, voice, voiceinfo, options)