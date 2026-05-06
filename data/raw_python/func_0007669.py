def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, character)`
        """
        for text_idx, text in enumerate(texts):
            if self.lower:
                text = text.lower()
            for char in text:
                yield text_idx, char