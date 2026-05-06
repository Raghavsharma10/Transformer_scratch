def unpad(text, block_size):
    """
    Takes the last character of the text, and if it is less than the block_size,
    assumes the text is padded, and removes any trailing zeros or bytes with the
    value of the pad character. See http://www.di-mgt.com.au/cryptopad.html for
    more information (methods 1, 3, and 4).
    """
    end = len(text)
    if end == 0:
        return text
    padch = ord_safe(text[end - 1])
    if padch > block_size:
        # If the last byte value is larger than the block size, it's not padded.
        return text
    while end > 0 and ord_safe(text[end - 1]) in (0, padch):
        end -= 1
    return text[:end]