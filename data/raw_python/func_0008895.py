def _calc_direction(data, mag, direction, ang, d1, d2, theta,
                    slc0, slc1, slc2):
    """
    This function gives the magnitude and direction of the slope based on
    Tarboton's D_\infty method. This is a helper-function to
    _tarboton_slopes_directions
    """
    
    data0 = data[slc0]
    data1 = data[slc1]
    data2 = data[slc2]

    s1 = (data0 - data1) / d1
    s2 = (data1 - data2) / d2
    s1_2 = s1**2
    
    sd = (data0 - data2) / np.sqrt(d1**2 + d2**2)
    r = np.arctan2(s2, s1)
    rad2 = s1_2 + s2**2

    # Handle special cases
    # should be on diagonal
    b_s1_lte0 = s1 <= 0
    b_s2_lte0 = s2 <= 0
    b_s1_gt0 = s1 > 0
    b_s2_gt0 = s2 > 0

    I1 = (b_s1_lte0 & b_s2_gt0) | (r > theta)
    if I1.any():
        rad2[I1] = sd[I1] ** 2
        r[I1] = theta.repeat(I1.shape[1], 1)[I1]
    
    I2 = (b_s1_gt0 & b_s2_lte0) | (r < 0)  # should be on straight section
    if I2.any():
        rad2[I2] = s1_2[I2]
        r[I2] = 0
    
    I3 = b_s1_lte0 & (b_s2_lte0 | (b_s2_gt0 & (sd <= 0)))  # upslope or flat
    rad2[I3] = -1

    I4 = rad2 > mag[slc0]
    if I4.any():
        mag[slc0][I4] = rad2[I4]
        direction[slc0][I4] = r[I4] * ang[1] + ang[0] * np.pi/2

    return mag, direction