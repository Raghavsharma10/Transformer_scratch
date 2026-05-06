def setup_data_split(X, y, tokenizer, proc_data_dir, **kwargs):
    """Setup data while splitting into a training, validation, and test set.

        Args:
            X: text data,
            y: data labels,
            tokenizer: A Tokenizer instance
            proc_data_dir: Directory for the split and processed data
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # only build vocabulary on training data
    tokenizer.build_vocab(X_train)

    process_save(X_train, y_train, tokenizer, path.join(
        proc_data_dir, 'train.bin'), train=True, **kwargs)
    process_save(X_val, y_val, tokenizer, path.join(
        proc_data_dir, 'val.bin'), **kwargs)
    process_save(X_test, y_test, tokenizer, path.join(
        proc_data_dir, 'test.bin'), **kwargs)