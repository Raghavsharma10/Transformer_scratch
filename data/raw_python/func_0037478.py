def match(apikey, pcmiter, samplerate, duration, channels=2, metadata=None):
    """Given a PCM data stream, perform fingerprinting and look up the
    metadata for the audio. pcmiter must be an iterable of blocks of
    PCM data (buffers). duration is the total length of the track in
    seconds (an integer). metadata may be a dictionary containing
    existing metadata for the file (optional keys: "artist", "album",
    and "title"). Returns a list of track info dictionaries
    describing the candidate metadata returned by Last.fm. Raises a
    subclass of FingerprintError if any step fails.
    """
    fpdata = extract(pcmiter, samplerate, channels)
    fpid = fpid_query(duration, fpdata, metadata)
    return metadata_query(fpid, apikey)