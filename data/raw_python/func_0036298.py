def doc_stream(path):
    """
    Generator to feed tokenized documents (treating each line as a document).
    """
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                yield line