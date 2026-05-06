def _add_new_labels(self, sentences):
        '''
        Adds new sentences to the internal indexing of the model.

        Args: 
            sentences (list): LabeledSentences for each doc to be added

        Returns:
            int: number of sentences added to the model

        '''
        sentence_no = -1
        total_words = 0
        vocab = self.model.vocab
        model_sentence_n = len([l for l in vocab if l.startswith("DOC_")])
        n_sentences = 0
        for sentence_no, sentence in enumerate(sentences):
            sentence_length = len(sentence.words)
            for label in sentence.labels:
                total_words += 1
                if label in vocab:
                    vocab[label].count += sentence_length
                else:
                    vocab[label] = gensim.models.word2vec.Vocab(
                        count=sentence_length)

                    vocab[label].index = len(self.model.vocab) - 1
                    vocab[label].code = [0]
                    vocab[label].sample_probability = 1.
                    self.model.index2word.append(label)
                    n_sentences += 1
                    
        return n_sentences