def to_wav(mediafile):
    """ Context manager providing a temporary WAV file created from the given media file.
    """
    if mediafile.endswith(".wav"):
        yield mediafile
    else:
        wavfile = tempfile.mktemp(__name__) + ".wav"
        try:
            extract_wav(mediafile, wavfile)
            yield wavfile
        finally:
            if os.path.exists(wavfile):
                os.remove(wavfile)