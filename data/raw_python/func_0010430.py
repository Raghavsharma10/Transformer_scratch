def get_engine_for_lang(self, lang):
        '''
        Determines the preferred engine/voice for a language.
        '''
        for eng in self.engines:
            if lang in eng.languages.keys():
                return eng
        raise TTSError('Could not match language')