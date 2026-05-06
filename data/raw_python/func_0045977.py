def trans_search(encoding):
    """Lookup transliterating codecs."""
    if encoding == 'transliterate':
        return codecs.CodecInfo(long_encode, no_decode)

    # translit/long/utf8
    # translit/one
    # translit/short/ascii

    if encoding.startswith('translit/'):
        parts = encoding.split('/')
        if parts[1] == 'long':
            encoder = long_encode
        elif parts[1] == 'short':
            encoder = short_encode
        elif parts[1] == 'one':
            encoder = single_encode
        else:
            return None

        if len(parts) == 2:
            pass
        elif len(parts) == 3:
            byte_enc = parts[2]
            byte_encoder = codecs.lookup(byte_enc).encode
            encoder = _double_encoding_factory(encoder, byte_encoder, byte_enc)
        else:
            return None
        return codecs.CodecInfo(encoder, no_decode)
    return None