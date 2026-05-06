def lemmatise(self, word):
        '''
        Tries to find the base form (lemma) of the given word, using
        the data provided by the Projekt deutscher Wortschatz.  This
        method returns a list of potential lemmas.

        >>> gn.lemmatise(u'Männer')
        [u'Mann']
        >>> gn.lemmatise(u'XYZ123')
        [u'XYZ123']
        '''
        lemmas = list(self._mongo_db.lemmatiser.find({'word': word}))
        if lemmas:
            return [lemma['lemma'] for lemma in lemmas]
        else:
            return [word]