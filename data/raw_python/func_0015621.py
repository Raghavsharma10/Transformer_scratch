def get_embedding_weights_from_file(word_dict, file_path, ignore_case=False):
    """Load pre-trained embeddings from a text file.

    Each line in the file should look like this:
        word feature_dim_1 feature_dim_2 ... feature_dim_n

    The `feature_dim_i` should be a floating point number.

    :param word_dict: A dict that maps words to indice.
    :param file_path: The location of the text file containing the pre-trained embeddings.
    :param ignore_case: Whether ignoring the case of the words.

    :return weights: A numpy array.
    """
    pre_trained = {}
    with codecs.open(file_path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if ignore_case:
                parts[0] = parts[0].lower()
            pre_trained[parts[0]] = list(map(float, parts[1:]))
    embd_dim = len(next(iter(pre_trained.values())))
    weights = [[0.0] * embd_dim for _ in range(max(word_dict.values()) + 1)]
    for word, index in word_dict.items():
        if not word:
            continue
        if ignore_case:
            word = word.lower()
        if word in pre_trained:
            weights[index] = pre_trained[word]
        else:
            weights[index] = numpy.random.random((embd_dim,)).tolist()
    return numpy.asarray(weights)