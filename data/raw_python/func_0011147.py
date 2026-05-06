def could_be_unfinished_char(seq, encoding):
    """Whether seq bytes might create a char in encoding if more bytes were added"""
    if decodable(seq, encoding):
        return False # any sensible encoding surely doesn't require lookahead (right?)
        # (if seq bytes encoding a character, adding another byte shouldn't also encode something)

    if encodings.codecs.getdecoder('utf8') is encodings.codecs.getdecoder(encoding):
        return could_be_unfinished_utf8(seq)
    elif encodings.codecs.getdecoder('ascii') is encodings.codecs.getdecoder(encoding):
        return False
    else:
        return True