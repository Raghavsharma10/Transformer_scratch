def get_decoder(encoding, flexible=False):
    """
    RETURN FUNCTION TO PERFORM DECODE
    :param encoding: STRING OF THE ENCODING
    :param flexible: True IF YOU WISH TO TRY OUR BEST, AND KEEP GOING
    :return: FUNCTION
    """
    if encoding == None:
        def no_decode(v):
            return v
        return no_decode
    elif flexible:
        def do_decode1(v):
            return v.decode(encoding, 'ignore')
        return do_decode1
    else:
        def do_decode2(v):
            return v.decode(encoding)
        return do_decode2