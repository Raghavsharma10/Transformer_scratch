def recort(input_data, surf, cortex):
    """
    Return data values to space of full cortex (including medial wall), with medial wall equal to zero.
    """
    data = np.zeros(len(surf[0]))
    data[cortex] = input_data
    return data