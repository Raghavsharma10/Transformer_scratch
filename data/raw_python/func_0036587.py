def vectorize( self, docs ):
        '''
        Returns the feature vectors for a set of docs. If model is not already be trained, 
        then self.train() is called.

        Args:
            docs (dict or list of tuples): asset_id, body_text of documents
            you wish to featurize.
        '''

        if type(docs) == dict:
            docs = docs.items()

        if self.model == None:
            self.train(docs)

        asset_id2vector = {}

        unfound = []
        for item in docs:
            ## iterate through the items in docs and check if any are already in the model.
            asset_id, _ = item
            label = 'DOC_' + str(asset_id)
            if label in self.model:
                asset_id2vector.update({asset_id: self.model['DOC_' + str(asset_id)]})
            else:
                unfound.append(item)

        if len(unfound) > 0:
            ## for all assets not in the model, update the model and then get their sentence vectors.
            sentences = [self._gen_sentence(item) for item in unfound]
            self.update_model(sentences, train=self.stream_train)
            asset_id2vector.update({item[0]: self.model['DOC_' + str(item[0])] for item in unfound})

        return asset_id2vector