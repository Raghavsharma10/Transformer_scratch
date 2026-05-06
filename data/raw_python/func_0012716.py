def is_stopword_record(self, record):
        """
        Determine whether a single MeCab record represents a stopword.

        This mostly determines words to strip based on their parts of speech.
        If common_words is set to True (default), it will also strip common
        verbs and nouns such as くる and よう. If more_stopwords is True, it
        will look at the sub-part of speech to remove more categories.
        """
        # preserve negations
        if record.root == 'ない':
            return False
        return (
            record.pos in STOPWORD_CATEGORIES or
            record.subclass1 in STOPWORD_CATEGORIES or
            record.root in STOPWORD_ROOTS
        )