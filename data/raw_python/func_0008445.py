def translate(self, from_lang=None, to="de"):
        """Translate the word to another language using Google's Translate API.

        .. versionadded:: 0.5.0 (``textblob``)

        """
        if from_lang is None:
            from_lang = self.translator.detect(self.string)
        return self.translator.translate(self.string,
                                         from_lang=from_lang, to_lang=to)