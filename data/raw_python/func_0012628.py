def get_velocity(samplemat, Hz, blinks=None):
    '''
    Compute velocity of eye-movements.

    Samplemat must contain fields 'x' and 'y', specifying the x,y coordinates
    of gaze location. The function assumes that the values in x,y are sampled
    continously at a rate specified by 'Hz'.
    '''
    Hz = float(Hz)
    distance = ((np.diff(samplemat.x) ** 2) +
                (np.diff(samplemat.y) ** 2)) ** .5
    distance = np.hstack(([distance[0]], distance))
    if blinks is not None:
        distance[blinks[1:]] = np.nan
    win = np.ones((velocity_window_size)) / float(velocity_window_size)
    velocity = np.convolve(distance, win, mode='same')
    velocity = velocity / (velocity_window_size / Hz)
    acceleration = np.diff(velocity) / (1. / Hz)
    acceleration = abs(np.hstack(([acceleration[0]], acceleration)))
    return velocity, acceleration