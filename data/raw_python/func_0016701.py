def synset(self, synset_repr):
        '''
        Looks up a synset in GermaNet using its string representation.

        Arguments:
        - `synset_repr`: a unicode string containing the lemma, part
          of speech, and sense number of the first lemma of the synset

        >>> gn.synset(u'funktionieren.v.2')
        Synset(funktionieren.v.2)
        '''
        parts = synset_repr.split('.')
        if len(parts) != 3:
            return None
        lemma, pos, sensenum = parts
        if not sensenum.isdigit() or pos not in SHORT_POS_TO_LONG:
            return None
        sensenum   = int(sensenum, 10)
        pos        = SHORT_POS_TO_LONG[pos]
        lemma_dict = self._mongo_db.lexunits.find_one({'orthForm': lemma,
                                                       'category': pos,
                                                       'sense':    sensenum})
        if lemma_dict:
            return Lemma(self, lemma_dict).synset