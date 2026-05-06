def write_to_human_readable(sdp, filename):
    """Write the SDP relaxation to a human-readable format.

    :param sdp: The SDP relaxation to write.
    :type sdp: :class:`ncpol2sdpa.sdp`.
    :param filename: The name of the file.
    :type filename: str.
    """
    objective, matrix = convert_to_human_readable(sdp)
    f = open(filename, 'w')
    f.write("Objective:" + objective + "\n")
    for matrix_line in matrix:
        f.write(str(list(matrix_line)).replace('[', '').replace(']', '')
                .replace('\'', ''))
        f.write('\n')
    f.close()