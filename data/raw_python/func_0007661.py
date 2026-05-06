def decode_texts(self, encoded_texts, unknown_token="<UNK>", inplace=True):
        """Decodes the texts using internal vocabulary. The list structure is maintained.

        Args:
            encoded_texts: The list of texts to decode.
            unknown_token: The placeholder value for unknown token. (Default value: "<UNK>")
            inplace: True to make changes inplace. (Default value: True)

        Returns:
            The decoded texts.
        """
        if len(self._token2idx) == 0:
            raise ValueError(
                "You need to build vocabulary using `build_vocab` before using `decode_texts`")

        if not isinstance(encoded_texts, list):
            # assume it's a numpy array
            encoded_texts = encoded_texts.tolist()

        if not inplace:
            encoded_texts = deepcopy(encoded_texts)
        utils._recursive_apply(encoded_texts,
                               lambda token_id: self._idx2token.get(token_id) or unknown_token)
        return encoded_texts