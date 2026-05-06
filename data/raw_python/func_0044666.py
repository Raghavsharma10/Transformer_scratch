def gradient_list(start, end, steps):
    """Compute gradient colors"""
    delta = [end[i] - start[i] for i in range(4)]
    return [bytearray(start[j] + (delta[j] * i) // steps for j in range(4))
            for i in range(steps + 1)]