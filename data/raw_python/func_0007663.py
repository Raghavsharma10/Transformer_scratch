def pad_sequences(self, sequences, fixed_sentences_seq_length=None, fixed_token_seq_length=None,
                      padding='pre', truncating='post', padding_token="<PAD>"):
        """Pads each sequence to the same fixed length (length of the longest sequence or provided override).

        Args:
            sequences: list of list (samples, words) or list of list of list (samples, sentences, words)
            fixed_sentences_seq_length: The fix sentence sequence length to use. If None, largest sentence length is used.
            fixed_token_seq_length: The fix token sequence length to use. If None, largest word length is used.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger than fixed_sentences_seq_length or fixed_token_seq_length
                either in the beginning or in the end of the sentence or word sequence respectively.
            padding_token: The token to add for padding.

        Returns:
            Numpy array of (samples, max_sentences, max_tokens) or (samples, max_tokens) depending on the sequence input.

        Raises:
            ValueError: in case of invalid values for `truncating` or `padding`.
        """
        value = self.special_token.index(padding_token)
        if value < 0:
            raise ValueError('The padding token "' + padding_token +
                             " is not in the special tokens of the tokenizer.")
        # Determine if input is (samples, max_sentences, max_tokens) or not.
        if isinstance(sequences[0][0], list):
            x = utils._pad_sent_sequences(sequences, fixed_sentences_seq_length,
                                          fixed_token_seq_length, padding, truncating, value)
        else:
            x = utils._pad_token_sequences(
                sequences, fixed_token_seq_length, padding, truncating, value)
        return np.array(x, dtype='int32')