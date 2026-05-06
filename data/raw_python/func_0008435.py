def sent_tokenize(self, text, **kwargs):
        """Returns a list of sentences.

        Each sentence is a space-separated string of tokens (words).
        Handles common cases of abbreviations (e.g., etc., ...).
        Punctuation marks are split from other words. Periods (or ?!) mark the end of a sentence.
        Headings without an ending period are inferred by line breaks.

        """

        sentences = find_sentences(text,
                                   punctuation=kwargs.get(
                                       "punctuation",
                                       PUNCTUATION),
                                   abbreviations=kwargs.get(
                                       "abbreviations",
                                       ABBREVIATIONS_DE),
                                   replace=kwargs.get("replace", replacements),
                                   linebreak=r"\n{2,}")
        return sentences