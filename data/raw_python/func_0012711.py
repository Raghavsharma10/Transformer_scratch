def tag_and_stem(self, text, cache=None):
        """
        Given some text, return a sequence of (stem, pos, text) triples as
        appropriate for the reader. `pos` can be as general or specific as
        necessary (for example, it might label all parts of speech, or it might
        only distinguish function words from others).

        Twitter-style hashtags and at-mentions have the stem and pos they would
        have without the leading # or @. For instance, if the reader's triple
        for "thing" is ('thing', 'NN', 'things'), then "#things" would come out
        as ('thing', 'NN', '#things').
        """
        analysis = self.analyze(text)
        triples = []

        for record in analysis:
            root = self.get_record_root(record)
            token = self.get_record_token(record)

            if token:
                if unicode_is_punctuation(token):
                    triples.append((token, '.', token))
                else:
                    pos = self.get_record_pos(record)
                    triples.append((root, pos, token))
        return triples