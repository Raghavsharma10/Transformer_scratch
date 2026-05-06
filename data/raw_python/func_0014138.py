def calculate_color_temperature(r, g, b):
    """Converts the raw R/G/B values to color temperature in degrees Kelvin."""
    # 1. Map RGB values to their XYZ counterparts.
    # Based on 6500K fluorescent, 3000K fluorescent
    # and 60W incandescent values for a wide range.
    # Note: Y = Illuminance or lux
    X = (-0.14282 * r) + (1.54924 * g) + (-0.95641 * b)
    Y = (-0.32466 * r) + (1.57837 * g) + (-0.73191 * b)
    Z = (-0.68202 * r) + (0.77073 * g) + ( 0.56332 * b)
    # Check for divide by 0 (total darkness) and return None.
    if (X + Y + Z) == 0:
        return None
    # 2. Calculate the chromaticity co-ordinates
    xc = (X) / (X + Y + Z)
    yc = (Y) / (X + Y + Z)
    # Check for divide by 0 again and return None.
    if (0.1858 - yc) == 0:
        return None
    # 3. Use McCamy's formula to determine the CCT
    n = (xc - 0.3320) / (0.1858 - yc)
    # Calculate the final CCT
    cct = (449.0 * (n ** 3.0)) + (3525.0 *(n ** 2.0)) + (6823.3 * n) + 5520.33
    return int(cct)