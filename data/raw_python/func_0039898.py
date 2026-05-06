def _process_cell(i, state, finite=False):
    """Process 3 cells and return a value from 0 to 7. """
    op_1 = state[i - 1]
    op_2 = state[i]
    if i == len(state) - 1:
        if finite:
            op_3 = state[0]
        else:
            op_3 = 0
    else:
        op_3 = state[i + 1]
    result = 0
    for i, val in enumerate([op_3, op_2, op_1]):
        if val:
            result += 2**i
    return result