def _phrase_doc_stream(paths, n, tokenizer=word_tokenize):
    """
    Generator to feed sentences to the phrase model.
    """
    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)
                for sent in sent_tokenize(line.lower()):
                    tokens = tokenizer(sent)
                    yield tokens