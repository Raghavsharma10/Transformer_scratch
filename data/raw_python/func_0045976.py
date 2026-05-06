def _double_encoding_factory(encoder, byte_encoder, byte_encoding):
    """Send the transliterated output to another codec."""
    def dbl_encode(input, errors='strict'):
        uni, length = encoder(input, errors)
        return byte_encoder(uni, errors)[0], length
    dbl_encode.__name__ = '%s_%s' % (encoder.__name__, byte_encoding)
    return dbl_encode