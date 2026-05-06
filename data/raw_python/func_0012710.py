def normalize_list(self, text, cache=None):
        """
        Get a canonical list representation of text, with words
        separated and reduced to their base forms.

        TODO: use the cache.
        """
        words = []
        analysis = self.analyze(text)
        for record in analysis:
            if not self.is_stopword_record(record):
                words.append(self.get_record_root(record))
        if not words:
            # Don't discard stopwords if that's all you've got
            words = [self.get_record_token(record) for record in analysis]
        return words