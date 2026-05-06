def classify(self, txt):
        '''
        Classifies text by language. Uses preferred_languages weighting.
        '''
        ranks = []
        for lang, score in langid.rank(txt):
            if lang in self.preferred_languages:
                score += self.preferred_factor
            ranks.append((lang, score))
        ranks.sort(key=lambda x: x[1], reverse=True)
        return ranks[0][0]