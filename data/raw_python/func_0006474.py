def pbkdf2_single(password, salt, key_length, prf):
    '''Returns the result of the Password-Based Key Derivation Function 2 with
       a single iteration (i.e. count = 1).

       prf - a psuedorandom function

       See http://en.wikipedia.org/wiki/PBKDF2
    '''

    block_number = 0
    result = b''

    # The iterations
    while len(result) < key_length:
        block_number += 1
        result += prf(password, salt + struct.pack('>L', block_number))

    return result[:key_length]