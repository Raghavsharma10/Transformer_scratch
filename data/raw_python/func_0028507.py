def translate(self, text, to_lang, from_lang=None, 
                content_type="text/plain", category=None):
        """
            This method takes as a parameter the desired text to be translated
            and the language to which should be translated. To find the code 
            for each language just go to the library home page.
            The parameter ::from_lang:: is optional because the api microsoft 
            recognizes the language used in a sentence automatically.
            The parameter ::content_type:: defaults to "text/plain". In fact
            it can be of two types: the very "text/plain" or "text/html".
            By default the parameter ::category:: is defined as "general", 
            we do not touch it.
        """
        infos_translate = TextModel(text, to_lang,
            from_lang, content_type, category).to_dict()
        mode_translate = TranslatorMode.Translate.value
        return self._get_content(infos_translate, mode_translate)