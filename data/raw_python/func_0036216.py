def train_doc2vec(paths, out='data/model.d2v', tokenizer=word_tokenize, sentences=False, **kwargs):
    """
    Train a doc2vec model on a list of files.
    """
    kwargs = {
        'size': 400,
        'window': 8,
        'min_count': 2,
        'workers': 8
    }.update(kwargs)

    n = 0
    for path in paths:
        print('Counting lines for {0}...'.format(path))
        n += sum(1 for line in open(path, 'r'))
    print('Processing {0} lines...'.format(n))

    print('Training doc2vec model...')
    m = Doc2Vec(_doc2vec_doc_stream(paths, n, tokenizer=tokenizer, sentences=sentences), **kwargs)

    print('Saving...')
    m.save(out)