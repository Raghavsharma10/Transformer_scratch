def match_file(apikey, path, metadata=None):
    """Uses the audioread library to decode an audio file and match it.
    """
    import audioread
    with audioread.audio_open(path) as f:
        return match(apikey, iter(f), f.samplerate, int(f.duration),
                     f.channels, metadata)