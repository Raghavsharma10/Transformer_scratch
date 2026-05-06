def add_doc(self, doc_id ='', doc_terms=[], doc_length=-1):
        '''
        Add new document to our Language Model (training phase)
        doc_id is used here, so we build seperate LF for each doc_id
        I.e. if you call it more than once with same doc_id,
        then all terms given via doc_terms will contribute to same LM
        doc_terms: list of words in document to be added 
        doc_length: the length of the document, you can provide it yourself,
                    otherwise, we use len(doc_terms) instead.
        '''
        if doc_length == -1:
            self.update_lengths(doc_id=doc_id, doc_length=len(doc_terms))
        else:
            self.update_lengths(doc_id=doc_id, doc_length=int(doc_length)) 
        for term in doc_terms: 
            self.vocabulary.add(term)
        terms = self.lr_padding(doc_terms)
        ngrams = self.to_ngrams(terms)    
        self.update_counts(doc_id, ngrams)