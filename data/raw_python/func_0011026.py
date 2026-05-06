def convert_to_human_readable(sdp):
    """Convert the SDP relaxation to a human-readable format.

    :param sdp: The SDP relaxation to write.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :returns: tuple of the objective function in a string and a matrix of
              strings as the symbolic representation of the moment matrix
    """

    objective = ""
    indices_in_objective = []
    for i, tmp in enumerate(sdp.obj_facvar):
        candidates = [key for key, v in
                      sdp.monomial_index.items() if v == i+1]
        if len(candidates) > 0:
            monomial = convert_monomial_to_string(candidates[0])
        else:
            monomial = ""
        if tmp > 0:
            objective += "+"+str(tmp)+monomial
            indices_in_objective.append(i)
        elif tmp < 0:
            objective += str(tmp)+monomial
            indices_in_objective.append(i)

    matrix_size = 0
    cumulative_sum = 0
    row_offsets = [0]
    block_offset = [0]
    for bs in sdp.block_struct:
        matrix_size += abs(bs)
        cumulative_sum += bs ** 2
        row_offsets.append(cumulative_sum)
        block_offset.append(matrix_size)

    matrix = []
    for i in range(matrix_size):
        matrix_line = ["0"] * matrix_size
        matrix.append(matrix_line)

    for row in range(len(sdp.F.rows)):
        if len(sdp.F.rows[row]) > 0:
            col_index = 0
            for k in sdp.F.rows[row]:
                value = sdp.F.data[row][col_index]
                col_index += 1
                block_index, i, j = convert_row_to_sdpa_index(
                    sdp.block_struct, row_offsets, row)
                candidates = [key for key, v in
                              sdp.monomial_index.items()
                              if v == k]
                if len(candidates) > 0:
                    monomial = convert_monomial_to_string(candidates[0])
                else:
                    monomial = ""
                offset = block_offset[block_index]
                if matrix[offset+i][offset+j] == "0":
                    matrix[offset+i][offset+j] = ("%s%s" % (value, monomial))
                else:
                    if value.real > 0:
                        matrix[offset+i][offset+j] += ("+%s%s" % (value,
                                                                  monomial))
                    else:
                        matrix[offset+i][offset+j] += ("%s%s" % (value,
                                                                 monomial))
    return objective, matrix