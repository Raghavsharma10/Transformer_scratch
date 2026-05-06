def find_amplitude(chunk):
    """
    Calculate the 0-1 amplitude of an ndarray chunk of audio samples.

    Samples in the ndarray chunk are signed int16 values oscillating
    anywhere between -32768 and 32767. Find the amplitude between 0 and 1
    by summing the absolute values of the minimum and maximum, and dividing
    by 32767.

    Args:
        chunk (numpy.ndarray): An array of int16 audio samples

    Returns:
        float: The amplitude of the sample between 0 and 1.
            Note that this is not a decibel representation of
            the amplitude.
    """
    return (abs(int(chunk.max() - chunk.min())) / config.SAMPLE_RANGE)