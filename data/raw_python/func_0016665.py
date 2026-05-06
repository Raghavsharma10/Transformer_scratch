def decode_qp(msg):
    """Decode SAPI response that uses `qp` format, without numpy.

    The 'qp' format is the current encoding used for problems and samples.
    In this encoding the reply is generally json, but the samples, energy,
    and histogram data (the occurrence count of each solution), are all
    base64 encoded arrays.
    """
    # Decode the simple buffers
    result = msg['answer']
    result['active_variables'] = _decode_ints(result['active_variables'])
    active_variables = result['active_variables']
    if 'num_occurrences' in result:
        result['num_occurrences'] = _decode_ints(result['num_occurrences'])
    result['energies'] = _decode_doubles(result['energies'])

    # Measure out the size of the binary solution data
    num_solutions = len(result['energies'])
    num_variables = len(result['active_variables'])
    solution_bytes = -(-num_variables // 8)  # equivalent to int(math.ceil(num_variables / 8.))
    total_variables = result['num_variables']

    # Figure out the null value for output
    default = 3 if msg['type'] == 'qubo' else 0

    # Decode the solutions, which will be byte aligned in binary format
    binary = base64.b64decode(result['solutions'])
    solutions = []
    for solution_index in range(num_solutions):
        # Grab the section of the buffer related to the current
        buffer_index = solution_index * solution_bytes
        solution_buffer = binary[buffer_index:buffer_index + solution_bytes]
        bytes = struct.unpack('B' * solution_bytes, solution_buffer)

        # Assume None values
        solution = [default] * total_variables
        index = 0
        for byte in bytes:
            # Parse each byte and read how ever many bits can be
            values = _decode_byte(byte)
            for _ in range(min(8, len(active_variables) - index)):
                i = active_variables[index]
                index += 1
                solution[i] = values.pop()

        # Switch to the right variable space
        if msg['type'] == 'ising':
            values = {0: -1, 1: 1}
            solution = [values.get(v, default) for v in solution]
        solutions.append(solution)

    result['solutions'] = solutions
    return result