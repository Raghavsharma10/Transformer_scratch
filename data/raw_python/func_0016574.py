def get_input_callback(samplerate, params, num_samples=256):
    """Return a function that produces samples of a sine.

    Parameters
    ----------
    samplerate : float
        The sample rate.
    params : dict
        Parameters for FM generation.
    num_samples : int, optional
        Number of samples to be generated on each call.
    """
    amplitude = params['mod_amplitude']
    frequency = params['mod_frequency']

    def producer():
        """Generate samples.

        Yields
        ------
        samples : ndarray
            A number of samples (`num_samples`) of the sine.
        """
        start_time = 0
        while True:
            time = start_time + np.arange(num_samples) / samplerate
            start_time += num_samples / samplerate
            output = amplitude * np.cos(2 * np.pi * frequency * time)
            yield output

    return lambda p=producer(): next(p)