def get_playback_callback(resampler, samplerate, params):
    """Return a sound playback callback.

    Parameters
    ----------
    resampler
        The resampler from which samples are read.
    samplerate : float
        The sample rate.
    params : dict
        Parameters for FM generation.
    """

    def callback(outdata, frames, time, _):
        """Playback callback.

        Read samples from the resampler and modulate them onto a carrier
        frequency.
        """
        last_fmphase = getattr(callback, 'last_fmphase', 0)
        df = params['fm_gain'] * resampler.read(frames)
        df = np.pad(df, (0, frames - len(df)), mode='constant')
        t = time.outputBufferDacTime + np.arange(frames) / samplerate
        phase = 2 * np.pi * params['carrier_frequency'] * t
        fmphase = last_fmphase + 2 * np.pi * np.cumsum(df) / samplerate
        outdata[:, 0] = params['output_volume'] * np.cos(phase + fmphase)
        callback.last_fmphase = fmphase[-1]

    return callback