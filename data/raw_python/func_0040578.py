def classify(self, text=u''):
        """ Predicts the Language of a given text.

            :param text: Unicode text to be classified.
        """
        result = self.calculate(doc_terms=self.tokenize(text))
        #return (result['calc_id'], result)
        return (result['calc_id'], self.karbasa(result))