def _doc2vec_doc_stream(paths, n, tokenizer=word_tokenize, sentences=True):
    """
    Generator to feed sentences to the dov2vec model.
    """
    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)

                # We do minimal pre-processing here so the model can learn
                # punctuation
                line = line.lower()

                if sentences:
                    for sent in sent_tokenize(line):
                        tokens = tokenizer(sent)
                        yield LabeledSentence(tokens, ['SENT_{}'.format(i)])
                else:
                    tokens = tokenizer(line)
                    yield LabeledSentence(tokens, ['SENT_{}'.format(i)])