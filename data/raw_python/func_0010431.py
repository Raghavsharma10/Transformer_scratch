def say(self, txt, lang=None):
        '''
        Says the text.

        if ``lang`` is ``None``, then uses ``classify()`` to detect language.
        '''
        lang = lang or self.classify(txt)
        self.get_engine_for_lang(lang).say(txt, language=lang)