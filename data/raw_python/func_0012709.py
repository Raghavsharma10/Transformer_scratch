def is_stopword(self, text):
        """
        Determine whether a single word is a stopword, or whether a short
        phrase is made entirely of stopwords, disregarding context.

        Use of this function should be avoided; it's better to give the text
        in context and let the process determine which words are the stopwords.
        """
        found_content_word = False
        for record in self.analyze(text):
            if not self.is_stopword_record(record):
                found_content_word = True
                break
        return not found_content_word