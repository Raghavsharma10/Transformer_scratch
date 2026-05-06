def interpolate_colors(array: numpy.ndarray, x: int) -> numpy.ndarray:
    """
    Creates a color map for values in array
    :param array: color map to interpolate
    :param x: number of colors
    :return: interpolated color map
    """
    out_array = []
    for i in range(x):
        if i % (x / (len(array) - 1)) == 0:
            index = i / (x / (len(array) - 1))
            out_array.append(array[int(index)])
        else:
            start_marker = array[math.floor(i / (x / (len(array) - 1)))]
            stop_marker = array[math.ceil(i / (x / (len(array) - 1)))]
            interp_amount = i % (x / (len(array) - 1)) / (x / (len(array) - 1))
            interp_color = numpy.rint(start_marker + ((stop_marker - start_marker) * interp_amount))
            out_array.append(interp_color)
    out_array[-1] = array[-1]
    return numpy.array(out_array).astype(numpy.uint8)