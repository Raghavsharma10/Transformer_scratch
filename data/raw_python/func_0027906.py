def tokenize(self):

        """
        Tokenize the text.
        """

        self.tokens = []
        self.terms = OrderedDict()

        # Generate tokens.
        for token in utils.tokenize(self.text):

            # Ignore stopwords.
            if token['unstemmed'] in self.stopwords:
                self.tokens.append(None)

            else:

                # Token:
                self.tokens.append(token)

                # Term:
                offsets = self.terms.setdefault(token['stemmed'], [])
                offsets.append(token['offset'])