def smoothstep(edge0, edge1, x):
    """ performs smooth Hermite interpolation
        between 0 and 1 when edge0 < x < edge1.  """
    # Scale, bias and saturate x to 0..1 range
    x = np.clip((x - edge0)/(edge1 - edge0), 0.0, 1.0)
    # Evaluate polynomial
    return x*x*(3 - 2*x)