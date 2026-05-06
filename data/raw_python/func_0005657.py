def uniorbytes(s, result=str, enc="utf-8", err="strict"):
    """
    This function was made to avoid byte / str type errors received in
    packages like base64. Accepts all input types and will recursively
    encode entire lists and dicts.

    @s: the #bytes or #str item you are attempting to encode or decode
    @result: the desired output, either #str or #bytes
    @enc: the desired encoding
    @err: passed to :meth:bytes.decode, tells the decoder what to do about
        errors, e.g. 'replace'

    -> type specified in @result
    """
    if isinstance(s, result):
        # the input is the desired one, return as is
        return s

    if isinstance(s, (bytes, str)):
        # the input is either a byte or a string, convert to desired
        # result (result=bytes or str)
        if isinstance(s, bytes) and result == str:
            return s.decode(enc, err)
        elif isinstance(s, str) and result == bytes:
            return s.encode(enc)
        else:
            return str(s or ("" if s is None else s), enc)
    elif isinstance(s, (float, int, decimal.Decimal)):
        return uniorbytes(str(s), result, enc, err)
    elif isinstance(s, dict):
        # the input is a dict {}
        for k, item in list(s.items()):
            s[k] = uniorbytes(item, result=result, enc=enc, err=err)
        return s
    elif hasattr(s, '__iter__'):
        # the input is iterable
        for i, item in enumerate(s):
            s[i] = uniorbytes(item, result=result, enc=enc, err=err)
        return s
    return s