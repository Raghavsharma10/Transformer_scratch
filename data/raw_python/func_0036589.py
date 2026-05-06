def update_model(self, sentences, update_labels_bool):
        '''
        takes a list of sentenes and updates an existing model. Vectors will be 
        callable through self.model[label]

        update_labels_bool: boolean that says whether to train the model (self.model.train_words = True)
        or simply to get vectors for the documents (self.model.train_words = False)

            self.vectorize should not train the model further
            self.train should if model already exists

        '''

        n_sentences = self._add_new_labels(sentences)

        # add new rows to self.model.syn0
        n = self.model.syn0.shape[0]
        self.model.syn0 = np.vstack((
            self.model.syn0,
            np.empty((n_sentences, self.model.layer1_size), dtype=np.float32)
        ))

        for i in xrange(n, n + n_sentences):
            np.random.seed(
                np.uint32(self.model.hashfxn(self.model.index2word[i] + str(self.model.seed))))
            a = (np.random.rand(self.model.layer1_size) - 0.5) / self.model.layer1_size
            self.model.syn0[i] = a

        # Set self.model.train_words to False and self.model.train_labels to True
        self.model.train_words = update_labels_bool
        self.model.train_lbls = True

        # train
        self.model.train(sentences)
        return