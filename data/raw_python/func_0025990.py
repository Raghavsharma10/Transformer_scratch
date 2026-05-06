def loadSignalFromWav(inputSignalFile, calibrationRealWorldValue=None, calibrationSignalFile=None, start=None,
                      end=None) -> Signal:
    """ reads a wav file into a Signal and scales the input so that the sample are expressed in real world values
    (as defined by the calibration signal).
    :param inputSignalFile: a path to the input signal file
    :param calibrationSignalFile: a path the calibration signal file
    :param calibrationRealWorldValue: the real world value represented by the calibration signal
    :param bitDepth: the bit depth of the input signal, used to rescale the value to a range of +1 to -1
    :returns: a Signal
    """
    inputSignal = readWav(inputSignalFile, start=start, end=end)
    if calibrationSignalFile is not None:
        calibrationSignal = readWav(calibrationSignalFile)
        scalingFactor = calibrationRealWorldValue / np.max(calibrationSignal.samples)
        return Signal(inputSignal.samples * scalingFactor, inputSignal.fs)
    else:
        return inputSignal