def classify(self, text=u''):
        """ Predicts the Language of a given text.

            :param text: Unicode text to be classified.
        """

        text = self.lm.normalize(text)
        tokenz = LM.tokenize(text, mode='c')
        result = self.lm.calculate(doc_terms=tokenz)
        #print 'Karbasa:', self.karbasa(result)
        if self.unk and self.lm.karbasa(result) < self.min_karbasa:
            lang = 'unk'
        else:
            lang = result['calc_id']
        return lang