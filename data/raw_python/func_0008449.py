def pos_tags(self):
        """Returns an list of tuples of the form (word, POS tag).

        Example:
        ::

            [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('on', 'IN'),
                    ('Thursday', 'NNP'), ('morning', 'NN')]

        :rtype: list of tuples

        """
        return [(Word(word, pos_tag=t), unicode(t))
                for word, t in self.pos_tagger.tag(self.raw)
                # new keyword PatternTagger(include_punc=False)
                # if not PUNCTUATION_REGEX.match(unicode(t))
                ]