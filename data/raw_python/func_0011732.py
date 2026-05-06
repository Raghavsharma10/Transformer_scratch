def _load_edflib(filename):
    """load a multi-channel Timeseries from an EDF (European Data Format) file
    or EDF+ file, using edflib.

    Args:
      filename: EDF+ file

    Returns:
      Timeseries
    """
    import edflib
    e = edflib.EdfReader(filename, annotations_mode='all')
    if np.ptp(e.get_samples_per_signal()) != 0:
        raise Error('channels have differing numbers of samples')
    if np.ptp(e.get_signal_freqs()) != 0:
        raise Error('channels have differing sample rates')
    n = e.samples_in_file(0)
    m = e.signals_in_file
    channelnames = e.get_signal_text_labels()
    dt = 1.0/e.samplefrequency(0)
    # EDF files hold <=16 bits of information for each sample. Representing as
    # double precision (64bit) is unnecessary use of memory. use 32 bit float:
    ar = np.zeros((n, m), dtype=np.float32)
    # edflib requires input buffer of float64s
    buf = np.zeros((n,), dtype=np.float64)
    for i in range(m):
        e.read_phys_signal(i, 0, n, buf)
        ar[:,i] = buf
    tspan = np.arange(0, (n - 1 + 0.5) * dt, dt, dtype=np.float32)
    return Timeseries(ar, tspan, labels=[None, channelnames])