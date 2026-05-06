def word_tokenize(self, text, include_punc=True):
        """The Treebank tokenizer uses regular expressions to tokenize text as
        in Penn Treebank.

        It assumes that the text has already been segmented into sentences,
        e.g. using ``self.sent_tokenize()``.

        This tokenizer performs the following steps:

        - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
        - treat most punctuation characters as separate tokens
        - split off commas and single quotes, when followed by whitespace
        - separate periods that appear at the end of line

        Source: NLTK's docstring of ``TreebankWordTokenizer`` (accessed: 02/10/2014)

        """
        #: Do not process empty strings (Issue #3)
        if text.strip() == "":
            return []
        _tokens = self.word_tok.tokenize(text)
        #: Handle strings consisting of a single punctuation mark seperately (Issue #4)
        if len(_tokens) == 1:
            if _tokens[0] in PUNCTUATION:
                if include_punc:
                    return _tokens
                else:
                    return []
        if include_punc:
            return _tokens
        else:
            # Return each word token
            # Strips punctuation unless the word comes from a contraction
            # e.g. "gibt's" => ["gibt", "'s"] in "Heute gibt's viel zu tun!"
            # e.g. "hat's" => ["hat", "'s"]
            # e.g. "home." => ['home']
            words = [
                word if word.startswith("'") else strip_punc(
                    word,
                    all=False) for word in _tokens if strip_punc(
                    word,
                    all=False)]
            return list(words)