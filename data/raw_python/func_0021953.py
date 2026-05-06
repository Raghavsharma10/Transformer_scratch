def concat(self, formatted_text):
        """:type formatted_text: FormattedText"""
        assert self._is_compatible(formatted_text), "Cannot concat text with different modes"
        self.text += formatted_text.text
        return self