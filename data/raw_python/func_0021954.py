def join(self, formatted_texts):
        """:type formatted_texts: list[FormattedText]"""
        formatted_texts = list(formatted_texts)  # so that after the first iteration elements are not lost if generator
        for formatted_text in formatted_texts:
            assert self._is_compatible(formatted_text), "Cannot join text with different modes"
        self.text = self.text.join((formatted_text.text for formatted_text in formatted_texts))
        return self