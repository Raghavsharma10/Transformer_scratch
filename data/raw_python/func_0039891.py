def to_ngrams(self, terms):
        ''' Converts terms to all possibe ngrams 
            terms: list of terms
        '''
        if len(terms) <= self.n:
            return terms
        if self.n == 1:
            n_grams = [[term] for term in terms]
        else:
            n_grams = []
            for i in range(0,len(terms)-self.n+1):
                n_grams.append(terms[i:i+self.n])
        return n_grams