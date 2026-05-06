def train(self, root=''):
        """ Trains our Language Model.

            :param root: Path to training data.
        """

        self.trainer = Train(root=root)
        corpus = self.trainer.get_corpus()

        # Show loaded Languages
        #print 'Lang Set: ' + ' '.join(train.get_lang_set())

        for item in corpus:
            self.lm.add_doc(doc_id=item[0], doc_terms=self._readfile(item[1]))

        # Save training timestamp
        self.training_timestamp = self.trainer.get_last_modified()