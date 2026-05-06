def encode_texts(self, texts, unknown_token="<UNK>", verbose=1, **kwargs):
        """Encodes the given texts using internal vocabulary with optionally applied encoding options. See
        ``apply_encoding_options` to set various options.

        Args:
            texts: The list of text items to encode.
            unknown_token: The token to replace words that out of vocabulary. If none, those words are omitted.
            verbose: The verbosity level for progress. Can be 0, 1, 2. (Default value = 1)
            **kwargs: The kwargs for `token_generator`.

        Returns:
            The encoded texts.
        """
        if not self.has_vocab:
            raise ValueError(
                "You need to build the vocabulary using `build_vocab` before using `encode_texts`")

        if unknown_token and unknown_token not in self.special_token:
            raise ValueError(
                "Your special token (" + unknown_token + ") to replace unknown words is not in the list of special token: " + self.special_token)

        progbar = Progbar(len(texts), verbose=verbose, interval=0.25)
        encoded_texts = []
        for token_data in self.token_generator(texts, **kwargs):
            indices, token = token_data[:-1], token_data[-1]

            token_idx = self._token2idx.get(token)
            if token_idx is None and unknown_token:
                token_idx = self.special_token.index(unknown_token)

            if token_idx is not None:
                utils._append(encoded_texts, indices, token_idx)

            # Update progressbar per document level.
            progbar.update(indices[0])

        # All done. Finalize progressbar.
        progbar.update(len(texts))
        return encoded_texts