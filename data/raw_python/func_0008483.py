def find_tokens(self, string, **kwargs):
        """ Returns a list of sentences from the given string.
            Punctuation marks are separated from each word by a space.
        """
        # "The cat purs." => ["The cat purs ."]
        return find_tokens(string,
                punctuation = kwargs.get(  "punctuation", PUNCTUATION),
              abbreviations = kwargs.get("abbreviations", ABBREVIATIONS),
                    replace = kwargs.get(      "replace", replacements),
                  linebreak = r"\n{2,}")