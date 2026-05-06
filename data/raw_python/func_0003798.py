def dot(r1, r2):
    """Compute the dot product

       Arguments:
        | ``r1``, ``r2``  -- two :class:`Vector3` objects

       (Returns a Scalar)
    """
    if r1.size != r2.size:
        raise ValueError("Both arguments must have the same input size.")
    if r1.deriv != r2.deriv:
        raise ValueError("Both arguments must have the same deriv.")
    return r1.x*r2.x + r1.y*r2.y + r1.z*r2.z