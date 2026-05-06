def blend(c1, c2):
    """Alpha blends two colors, using the alpha given by c2"""
    return [c1[i] * (0xFF - c2[3]) + c2[i] * c2[3] >> 8 for i in range(3)]