def extract_dual_value(sdp, monomial, blocks=None):
    """Given a solution of the dual problem and a monomial, it returns the
    inner product of the corresponding coefficient matrix and the dual
    solution. It can be restricted to certain blocks.

    :param sdp: The SDP relaxation.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param monomial: The monomial for which the value is requested.
    :type monomial: :class:`sympy.core.exp.Expr`.
    :param monomial: The monomial for which the value is requested.
    :type monomial: :class:`sympy.core.exp.Expr`.
    :param blocks: Optional parameter to specify the blocks to be included.
    :type blocks: list of `int`.
    :returns: The value of the monomial in the solved relaxation.
    :rtype: float.
    """
    if sdp.status == "unsolved":
        raise Exception("The SDP relaxation is unsolved!")
    if blocks is None:
        blocks = [i for i, _ in enumerate(sdp.block_struct)]
    if is_number_type(monomial):
        index = 0
    else:
        index = sdp.monomial_index[monomial]
    row_offsets = [0]
    cumulative_sum = 0
    for block_size in sdp.block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    result = 0
    for row in range(len(sdp.F.rows)):
        if len(sdp.F.rows[row]) > 0:
            col_index = 0
            for k in sdp.F.rows[row]:
                if k != index:
                    continue
                value = sdp.F.data[row][col_index]
                col_index += 1
                block_index, i, j = convert_row_to_sdpa_index(
                    sdp.block_struct, row_offsets, row)
                if block_index in blocks:
                    result += -value*sdp.y_mat[block_index][i][j]
    return result