def setup_data(X, y, tokenizer, proc_data_path, **kwargs):
    """Setup data

        Args:
            X: text data,
            y: data labels,
            tokenizer: A Tokenizer instance
            proc_data_path: Path for the processed data
    """
    # only build vocabulary once (e.g. training data)
    train = not tokenizer.has_vocab
    if train:
        tokenizer.build_vocab(X)

    process_save(X, y, tokenizer, proc_data_path,
                 train=train, **kwargs)
    return tokenizer