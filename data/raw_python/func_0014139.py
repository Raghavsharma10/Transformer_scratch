def calculate_lux(r, g, b):
    """Converts the raw R/G/B values to luminosity in lux."""
    illuminance = (-0.32466 * r) + (1.57837 * g) + (-0.73191 * b)
    return int(illuminance)