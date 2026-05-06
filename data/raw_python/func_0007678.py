def build_model(self, token_encoder_model, trainable_embeddings=True, output_activation='softmax'):
        """Builds a model using the given `text_model`

        Args:
            token_encoder_model: An instance of `SequenceEncoderBase` for encoding all the tokens within a document.
                This encoding is then fed into a final `Dense` layer for classification.
            trainable_embeddings: Whether or not to fine tune embeddings.
            output_activation: The output activation to use. (Default value: 'softmax')
                Use:
                - `softmax` for binary or multi-class.
                - `sigmoid` for multi-label classification.
                - `linear` for regression output.

        Returns:
            The model output tensor.
        """
        if not isinstance(token_encoder_model, SequenceEncoderBase):
            raise ValueError("`token_encoder_model` should be an instance of `{}`".format(
                SequenceEncoderBase))

        if not token_encoder_model.allows_dynamic_length() and self.max_tokens is None:
            raise ValueError("The provided `token_encoder_model` does not allow variable length mini-batches. "
                             "You need to provide `max_tokens`")

        if self.embeddings_index is None:
            # The +1 is for unknown token index 0.
            embedding_layer = Embedding(len(self.token_index),
                                        self.embedding_dims,
                                        input_length=self.max_tokens,
                                        mask_zero=token_encoder_model.allows_dynamic_length(),
                                        trainable=trainable_embeddings)
        else:
            embedding_layer = Embedding(len(self.token_index),
                                        self.embedding_dims,
                                        weights=[build_embedding_weights(
                                            self.token_index, self.embeddings_index)],
                                        input_length=self.max_tokens,
                                        mask_zero=token_encoder_model.allows_dynamic_length(),
                                        trainable=trainable_embeddings)

        sequence_input = Input(shape=(self.max_tokens,), dtype='int32')
        x = embedding_layer(sequence_input)
        x = token_encoder_model(x)
        x = Dense(self.num_classes, activation=output_activation)(x)
        return Model(sequence_input, x)