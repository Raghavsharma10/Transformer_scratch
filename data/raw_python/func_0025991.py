def readWav(inputSignalFile, selectedChannel=1, start=None, end=None) -> Signal:
    """ reads a wav file into a Signal.
    :param inputSignalFile: a path to the input signal file
    :param selectedChannel: the channel to read.
    :param start: the time to start reading from in HH:mm:ss.SSS format.
    :param end: the time to end reading from in HH:mm:ss.SSS format.
    :returns: Signal.
    """

    def asFrames(time, fs):
        hours, minutes, seconds = (time.split(":"))[-3:]
        hours = int(hours)
        minutes = int(minutes)
        seconds = float(seconds)
        millis = int((3600000 * hours) + (60000 * minutes) + (1000 * seconds))
        return int(millis * (fs / 1000))

    import soundfile as sf
    if start is not None or end is not None:
        info = sf.info(inputSignalFile)
        startFrame = 0 if start is None else asFrames(start, info.samplerate)
        endFrame = None if end is None else asFrames(end, info.samplerate)
        ys, frameRate = sf.read(inputSignalFile, start=startFrame, stop=endFrame)
    else:
        ys, frameRate = sf.read(inputSignalFile)
    return Signal(ys[::selectedChannel], frameRate)