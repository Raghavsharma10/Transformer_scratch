def segment_array(arr, length, overlap=.5):
    """
    Segment array into chunks of a specified length, with a specified
    proportion overlap.

    Operates on axis 0.

    :param integer length: Length of each segment
    :param float overlap: Proportion overlap of each frame
    """

    arr = N.array(arr)

    offset = float(overlap) * length
    total_segments = int((N.shape(arr)[0] - length) / offset) + 1
    # print "total segments", total_segments

    other_shape = N.shape(arr)[1:]
    out_shape = [total_segments, length]
    out_shape.extend(other_shape)

    out = N.empty(out_shape)

    for i in xrange(total_segments):
        out[i][:] = arr[i * offset:i * offset + length]

    return out