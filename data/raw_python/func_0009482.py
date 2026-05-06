def tiltFactor(xy, f, tilt, rot, center=None):
    '''
    this function is extra to only cover vignetting through perspective distortion

    f - focal length [px]
    tau - tilt angle of a planar scene [radian]
    rot - rotation angle of a planar scene [radian]
    '''
    x, y = xy
    arr = np.cos(tilt) * (
        1 + (np.tan(tilt) / f) * (
            x * np.sin(rot) - y * np.cos(rot)))**3
    return arr