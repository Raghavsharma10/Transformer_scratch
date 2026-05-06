def aes_pad(s, block_size=32, padding='{'):
    """ Adds padding to get the correct block sizes for AES encryption

        @s: #str being AES encrypted or decrypted
        @block_size: the AES block size
        @padding: character to pad with

        -> padded #str

        ..
            from vital.security import aes_pad
            aes_pad("swing")
            # -> 'swing{{{{{{{{{{{{{{{{{{{{{{{{{{{'
        ..
    """
    return s + (block_size - len(s) % block_size) * padding