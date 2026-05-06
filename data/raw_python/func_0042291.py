def ibytes2ilines(generator, encoding="utf8", flexible=False, closer=None):
    """
    CONVERT A GENERATOR OF (ARBITRARY-SIZED) byte BLOCKS
    TO A LINE (CR-DELIMITED) GENERATOR

    :param generator:
    :param encoding: None TO DO NO DECODING
    :param closer: OPTIONAL FUNCTION TO RUN WHEN DONE ITERATING
    :return:
    """
    decode = get_decoder(encoding=encoding, flexible=flexible)
    _buffer = generator.next()
    s = 0
    e = _buffer.find(b"\n")
    while True:
        while e == -1:
            try:
                next_block = generator.next()
                _buffer = _buffer[s:] + next_block
                s = 0
                e = _buffer.find(b"\n")
            except StopIteration:
                _buffer = _buffer[s:]
                del generator
                if closer:
                    closer()
                if _buffer:
                    yield decode(_buffer)
                return

        yield decode(_buffer[s:e])
        s = e + 1
        e = _buffer.find(b"\n", s)