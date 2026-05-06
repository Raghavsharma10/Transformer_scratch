def get_embedding_layer(word_dict_len,
                        char_dict_len,
                        max_word_len,
                        word_embd_dim=300,
                        char_embd_dim=30,
                        char_hidden_dim=150,
                        char_hidden_layer_type='lstm',
                        word_embd_weights=None,
                        char_embd_weights=None,
                        word_embd_trainable=None,
                        char_embd_trainable=None,
                        word_mask_zero=True,
                        char_mask_zero=True):
    """Get the merged embedding layer.

    :param word_dict_len: The number of words in the dictionary including the ones mapped to 0 or 1.
    :param char_dict_len: The number of characters in the dictionary including the ones mapped to 0 or 1.
    :param max_word_len: The maximum allowed length of word.
    :param word_embd_dim: The dimensions of the word embedding.
    :param char_embd_dim: The dimensions of the character embedding
    :param char_hidden_dim: The dimensions of the hidden states of RNN in one direction.
    :param word_embd_weights: A numpy array representing the pre-trained embeddings for words.
    :param char_embd_weights: A numpy array representing the pre-trained embeddings for characters.
    :param word_embd_trainable: Whether the word embedding layer is trainable.
    :param char_embd_trainable: Whether the character embedding layer is trainable.
    :param char_hidden_layer_type: The type of the recurrent layer, 'lstm' or 'gru'.
    :param word_mask_zero: Whether enable the mask for words.
    :param char_mask_zero: Whether enable the mask for characters.

    :return inputs, embd_layer: The keras layer.
    """
    if word_embd_weights is not None:
        word_embd_weights = [word_embd_weights]
    if word_embd_trainable is None:
        word_embd_trainable = word_embd_weights is None

    if char_embd_weights is not None:
        char_embd_weights = [char_embd_weights]
    if char_embd_trainable is None:
        char_embd_trainable = char_embd_weights is None

    word_input_layer = keras.layers.Input(
        shape=(None,),
        name='Input_Word',
    )
    char_input_layer = keras.layers.Input(
        shape=(None, max_word_len),
        name='Input_Char',
    )

    word_embd_layer = keras.layers.Embedding(
        input_dim=word_dict_len,
        output_dim=word_embd_dim,
        mask_zero=word_mask_zero,
        weights=word_embd_weights,
        trainable=word_embd_trainable,
        name='Embedding_Word',
    )(word_input_layer)
    char_embd_layer = keras.layers.Embedding(
        input_dim=char_dict_len,
        output_dim=char_embd_dim,
        mask_zero=char_mask_zero,
        weights=char_embd_weights,
        trainable=char_embd_trainable,
        name='Embedding_Char_Pre',
    )(char_input_layer)
    if char_hidden_layer_type == 'lstm':
        char_hidden_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=char_hidden_dim,
                input_shape=(max_word_len, char_dict_len),
                return_sequences=False,
                return_state=False,
            ),
            name='Bi-LSTM_Char',
        )
    elif char_hidden_layer_type == 'gru':
        char_hidden_layer = keras.layers.Bidirectional(
            keras.layers.GRU(
                units=char_hidden_dim,
                input_shape=(max_word_len, char_dict_len),
                return_sequences=False,
                return_state=False,
            ),
            name='Bi-GRU_Char',
        )
    elif char_hidden_layer_type == 'cnn':
        char_hidden_layer = [
            MaskedConv1D(
                filters=max(1, char_hidden_dim // 5),
                kernel_size=3,
                activation='relu',
            ),
            MaskedFlatten(),
            keras.layers.Dense(
                units=char_hidden_dim,
                name='Dense_Char',
            ),
        ]
    elif isinstance(char_hidden_layer_type, list) or isinstance(char_hidden_layer_type, keras.layers.Layer):
        char_hidden_layer = char_hidden_layer_type
    else:
        raise NotImplementedError('Unknown character hidden layer type: %s' % char_hidden_layer_type)
    if not isinstance(char_hidden_layer, list):
        char_hidden_layer = [char_hidden_layer]
    for i, layer in enumerate(char_hidden_layer):
        if i == len(char_hidden_layer) - 1:
            name = 'Embedding_Char'
        else:
            name = 'Embedding_Char_Pre_%d' % (i + 1)
        char_embd_layer = keras.layers.TimeDistributed(layer=layer, name=name)(char_embd_layer)
    embd_layer = keras.layers.Concatenate(
        name='Embedding',
    )([word_embd_layer, char_embd_layer])
    return [word_input_layer, char_input_layer], embd_layer