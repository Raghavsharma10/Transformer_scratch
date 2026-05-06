def angle2vecs(vec1, vec2):
    """angle between two vectors"""
    # vector a * vector b = |a|*|b|* cos(angle between vector a and vector b)
    dot = np.dot(vec1, vec2)
    vec1_modulus = np.sqrt(np.multiply(vec1, vec1).sum())
    vec2_modulus = np.sqrt(np.multiply(vec2, vec2).sum())
    if (vec1_modulus * vec2_modulus) == 0:
        cos_angle = 1
    else: cos_angle = dot / (vec1_modulus * vec2_modulus)
    return math.degrees(acos(cos_angle))