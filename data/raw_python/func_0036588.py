def train(self, docs, retrain=False):
        '''
        Train Doc2Vec on a series of docs. Train from scratch or update.

        Args:
            docs: list of tuples (assetid, body_text) or dictionary {assetid : body_text}
            retrain: boolean, retrain from scratch or update model

        saves model in class to self.model   

        Returns: 0 if successful
        '''

        if type(docs) == dict:
            docs = docs.items()

        train_sentences = [self._gen_sentence(item) for item in docs]
        if (self.is_trained) and (retrain == False): 
            ## online training 
            self.update_model(train_sentences, update_labels_bool=True)

        else: 
            ## train from scratch
            self.model = Doc2Vec(train_sentences, size=self.size, window=self.window, min_count=self.min_count, workers=self.workers)
            self.is_trained = True

        return 0