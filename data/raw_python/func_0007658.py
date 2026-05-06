def create_token_indices(self, tokens):
        """If `apply_encoding_options` is inadequate, one can retrieve tokens from `self.token_counts`, filter with
        a desired strategy and regenerate `token_index` using this method. The token index is subsequently used
        when `encode_texts` or `decode_texts` methods are called.
        """
        start_index = len(self.special_token)
        indices = list(range(len(tokens) + start_index))
        # prepend because the special tokens come in the beginning
        tokens_with_special = self.special_token + list(tokens)
        self._token2idx = dict(list(zip(tokens_with_special, indices)))
        self._idx2token = dict(list(zip(indices, tokens_with_special)))