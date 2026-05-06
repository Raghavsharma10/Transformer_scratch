def _merge_sampleset(model1, model2):
    """Simple merge of samplesets."""
    w1 = _get_sampleset(model1)
    w2 = _get_sampleset(model2)
    return merge_wavelengths(w1, w2)