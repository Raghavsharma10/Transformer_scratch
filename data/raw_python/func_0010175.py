def nearly_eq(valA, valB, maxf=None, minf=None, epsilon=None):
    '''
    implementation based on:

    http://floating-point-gui.de/errors/comparison/
    '''

    if valA == valB:
        return True

    if maxf is None:
        maxf = float_info.max

    if minf is None:
        minf = float_info.min

    if epsilon is None:
        epsilon = float_info.epsilon

    absA = abs(valA)
    absB = abs(valB)
    delta = abs(valA - valB)

    if (valA == 0) or (valB == 0) or (delta < minf):
        return delta < (epsilon * minf)

    return (delta / min(absA + absB, maxf)) < (epsilon * 2)