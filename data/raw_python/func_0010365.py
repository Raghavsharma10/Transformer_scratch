def detect(filename, include_confidence=False):
    """
    Detect the encoding of a file.

    Returns only the predicted current encoding as a string.

    If `include_confidence` is True, 
    Returns tuple containing: (str encoding, float confidence)
    """
    f = open(filename)
    detection = chardet.detect(f.read())
    f.close()
    encoding = detection.get('encoding')
    confidence = detection.get('confidence')
    if include_confidence:
        return (encoding, confidence)
    return encoding