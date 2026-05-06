def subCell2DCoords(*args, **kwargs):
    '''Same as subCell2DSlices but returning coordinates

    Example:
        g = subCell2DCoords(arr, shape)
        for x, y in g:
                plt.plot(x, y)
    '''
    for _, _, s0, s1 in subCell2DSlices(*args, **kwargs):
        yield ((s1.start, s1.start, s1.stop),
               (s0.start, s0.stop,  s0.stop))