def voice(self):
        """tuple. contain text and lang code
        """
        dbid = self.lldb.dbid
        text, lang = self._voiceoverdb.get_text_lang(dbid)
        return text, lang