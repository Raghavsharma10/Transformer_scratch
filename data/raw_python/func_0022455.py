def docs2vecs(self, docs):
        """Convert multiple SimilarityDocuments to vectors (batch version of doc2vec)."""
        bows = (self.dictionary.doc2bow(doc['tokens']) for doc in docs)
        if self.method == 'lsi':
            return self.lsi[self.tfidf[bows]]
        elif self.method == 'lda':
            return self.lda[bows]
        elif self.method == 'lda_tfidf':
            return self.lda[self.tfidf[bows]]
        elif self.method == 'logentropy':
            return self.logent[bows]