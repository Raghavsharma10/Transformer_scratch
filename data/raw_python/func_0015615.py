def get_embedding_layer(self,
                            word_embd_dim=300,
                            char_embd_dim=30,
                            char_hidden_dim=150,
                            char_hidden_layer_type='lstm',
                            word_embd_weights=None,
                            word_embd_file_path=None,
                            char_embd_weights=None,
                            char_embd_file_path=None,
                            word_embd_trainable=None,
                            char_embd_trainable=None,
                            word_mask_zero=True,
                            char_mask_zero=True,):
        """Get the merged embedding layer.

        :param word_embd_dim: The dimensions of the word embedding.
        :param char_embd_dim: The dimensions of the character embedding
        :param char_hidden_dim: The dimensions of the hidden states of RNN in one direction.
        :param word_embd_weights: A numpy array representing the pre-trained embeddings for words.
        :param word_embd_file_path: The file that contains the word embeddings.
        :param char_embd_weights: A numpy array representing the pre-trained embeddings for characters.
        :param char_embd_file_path: The file that contains the character embeddings.
        :param word_embd_trainable: Whether the word embedding layer is trainable.
        :param char_embd_trainable: Whether the character embedding layer is trainable.
        :param char_hidden_layer_type: The type of the recurrent layer, 'lstm' or 'gru'.
        :param word_mask_zero: Whether enable the mask for words.
        :param char_mask_zero: Whether enable the mask for characters.

        :return inputs, embd_layer: The keras layer.
        """
        if word_embd_file_path is not None:
            word_embd_weights = get_embedding_weights_from_file(word_dict=self.get_word_dict(),
                                                                file_path=word_embd_file_path,
                                                                ignore_case=self.word_ignore_case)
        if char_embd_file_path is not None:
            char_embd_weights = get_embedding_weights_from_file(word_dict=self.get_char_dict(),
                                                                file_path=char_embd_file_path,
                                                                ignore_case=self.char_ignore_case)
        return get_embedding_layer(word_dict_len=len(self.get_word_dict()),
                                   char_dict_len=len(self.get_char_dict()),
                                   max_word_len=self.max_word_len,
                                   word_embd_dim=word_embd_dim,
                                   char_embd_dim=char_embd_dim,
                                   char_hidden_dim=char_hidden_dim,
                                   char_hidden_layer_type=char_hidden_layer_type,
                                   word_embd_weights=word_embd_weights,
                                   char_embd_weights=char_embd_weights,
                                   word_embd_trainable=word_embd_trainable,
                                   char_embd_trainable=char_embd_trainable,
                                   word_mask_zero=word_mask_zero,
                                   char_mask_zero=char_mask_zero)