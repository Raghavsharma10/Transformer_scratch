def pad(text, block_size, zero=False):
    """
    Given a text string and a block size, pads the text with bytes of the same value
    as the number of padding bytes. This is the recommended method, and the one used
    by pgcrypto. See http://www.di-mgt.com.au/cryptopad.html for more information.
    """
    num = block_size - (len(text) % block_size)
    ch = b'\0' if zero else chr(num).encode('latin-1')
    return text + (ch * num)