def _sc_encode(gain, peak):
    """Encode ReplayGain gain/peak values as a Sound Check string.
    """
    # SoundCheck stores the peak value as the actual value of the
    # sample, rather than the percentage of full scale that RG uses, so
    # we do a simple conversion assuming 16 bit samples.
    peak *= 32768.0

    # SoundCheck stores absolute RMS values in some unknown units rather
    # than the dB values RG uses. We can calculate these absolute values
    # from the gain ratio using a reference value of 1000 units. We also
    # enforce the maximum value here, which is equivalent to about
    # -18.2dB.
    g1 = int(min(round((10 ** (gain / -10)) * 1000), 65534))
    # Same as above, except our reference level is 2500 units.
    g2 = int(min(round((10 ** (gain / -10)) * 2500), 65534))

    # The purpose of these values are unknown, but they also seem to be
    # unused so we just use zero.
    uk = 0
    values = (g1, g1, g2, g2, uk, uk, int(peak), int(peak), uk, uk)
    return (u' %08X' * 10) % values