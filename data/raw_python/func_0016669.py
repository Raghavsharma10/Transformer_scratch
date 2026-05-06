def decode_qp_numpy(msg, return_matrix=True):
    """Decode SAPI response, results in a `qp` format, explicitly using numpy.
    If numpy is not installed, the method will fail.

    To use numpy for decoding, but return the results a lists (instead of
    numpy matrices), set `return_matrix=False`.
    """
    import numpy as np

    result = msg['answer']

    # Build some little endian type encodings
    double_type = np.dtype(np.double)
    double_type = double_type.newbyteorder('<')
    int_type = np.dtype(np.int32)
    int_type = int_type.newbyteorder('<')

    # Decode the simple buffers
    result['energies'] = np.frombuffer(base64.b64decode(result['energies']),
                                       dtype=double_type)

    if 'num_occurrences' in result:
        result['num_occurrences'] = \
            np.frombuffer(base64.b64decode(result['num_occurrences']),
                        dtype=int_type)

    result['active_variables'] = \
        np.frombuffer(base64.b64decode(result['active_variables']),
                      dtype=int_type)

    # Measure out the binary data size
    num_solutions = len(result['energies'])
    active_variables = result['active_variables']
    num_variables = len(active_variables)
    total_variables = result['num_variables']

    # Decode the solutions, which will be a continuous run of bits
    byte_type = np.dtype(np.uint8)
    byte_type = byte_type.newbyteorder('<')
    bits = np.unpackbits(np.frombuffer(base64.b64decode(result['solutions']),
                         dtype=byte_type))

    # Clip off the extra bits from encoding
    if num_solutions:
        bits = np.reshape(bits, (num_solutions, bits.size // num_solutions))
        bits = np.delete(bits, range(num_variables, bits.shape[1]), 1)

    # Switch from bits to spins
    default = 3
    if msg['type'] == 'ising':
        bits = bits.astype(np.int8)
        bits *= 2
        bits -= 1
        default = 0

    # Fill in the missing variables
    solutions = np.full((num_solutions, total_variables), default, dtype=np.int8)
    solutions[:, active_variables] = bits
    result['solutions'] = solutions

    # If the final result shouldn't be numpy formats switch back to python objects
    if not return_matrix:
        result['energies'] = result['energies'].tolist()
        if 'num_occurrences' in result:
            result['num_occurrences'] = result['num_occurrences'].tolist()
        result['active_variables'] = result['active_variables'].tolist()
        result['solutions'] = result['solutions'].tolist()

    return result