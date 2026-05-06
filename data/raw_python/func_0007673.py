def process_save(X, y, tokenizer, proc_data_path, max_len=400, train=False, ngrams=None, limit_top_tokens=None):
    """Process text and save as Dataset
    """
    if train and limit_top_tokens is not None:
        tokenizer.apply_encoding_options(limit_top_tokens=limit_top_tokens)

    X_encoded = tokenizer.encode_texts(X)

    if ngrams is not None:
        X_encoded = tokenizer.add_ngrams(X_encoded, n=ngrams, train=train)

    X_padded = tokenizer.pad_sequences(
        X_encoded, fixed_token_seq_length=max_len)

    if train:
        ds = Dataset(X_padded,
                     y, tokenizer=tokenizer)
    else:
        ds = Dataset(X_padded, y)

    ds.save(proc_data_path)