def build_vocab(self, texts, verbose=1, **kwargs):
        """Builds the internal vocabulary and computes various statistics.

        Args:
            texts: The list of text items to encode.
            verbose: The verbosity level for progress. Can be 0, 1, 2. (Default value = 1)
            **kwargs: The kwargs for `token_generator`.
        """
        if self.has_vocab:
            logger.warn(
                "Tokenizer already has existing vocabulary. Overriding and building new vocabulary.")

        progbar = Progbar(len(texts), verbose=verbose, interval=0.25)
        count_tracker = utils._CountTracker()

        self._token_counts.clear()
        self._num_texts = len(texts)

        for token_data in self.token_generator(texts, **kwargs):
            indices, token = token_data[:-1], token_data[-1]
            count_tracker.update(indices)
            self._token_counts[token] += 1

            # Update progressbar per document level.
            progbar.update(indices[0])

        # Generate token2idx and idx2token.
        self.create_token_indices(self._token_counts.keys())

        # All done. Finalize progressbar update and count tracker.
        count_tracker.finalize()
        self._counts = count_tracker.counts
        progbar.update(len(texts))