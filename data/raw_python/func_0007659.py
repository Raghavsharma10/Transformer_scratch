def apply_encoding_options(self, min_token_count=1, limit_top_tokens=None):
        """Applies the given settings for subsequent calls to `encode_texts` and `decode_texts`. This allows you to
        play with different settings without having to re-run tokenization on the entire corpus.

        Args:
            min_token_count: The minimum token count (frequency) in order to include during encoding. All tokens
                below this frequency will be encoded to `0` which corresponds to unknown token. (Default value = 1)
            limit_top_tokens: The maximum number of tokens to keep, based their frequency. Only the most common `limit_top_tokens`
                tokens will be kept. Set to None to keep everything. (Default value: None)
        """
        if not self.has_vocab:
            raise ValueError("You need to build the vocabulary using `build_vocab` "
                             "before using `apply_encoding_options`")
        if min_token_count < 1:
            raise ValueError("`min_token_count` should atleast be 1")

        # Remove tokens with freq < min_token_count
        token_counts = list(self._token_counts.items())
        token_counts = [x for x in token_counts if x[1] >= min_token_count]

        # Clip to max_tokens.
        if limit_top_tokens is not None:
            token_counts.sort(key=lambda x: x[1], reverse=True)
            filtered_tokens = list(zip(*token_counts))[0]
            filtered_tokens = filtered_tokens[:limit_top_tokens]
        else:
            filtered_tokens = zip(*token_counts)[0]

        # Generate indices based on filtered tokens.
        self.create_token_indices(filtered_tokens)