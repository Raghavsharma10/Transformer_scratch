def oscillating_setpoint(_square_wave=False, shift=0):
    """A basic example of a target that you may want to approximate.

    If you have a thermostat, this is a temperature setting.
    This target can't change too often
    """
    import math
    c = 0
    while 1:
        if _square_wave:
            yield ((c % 300) < 150) * 30 + 20
            c += 1
        else:
            yield 10 * math.sin(2 * 3.1415926 * c + shift) \
                + 20 + 5 * math.sin(2 * 3.1415926 * c * 3 + shift)
            c += .001