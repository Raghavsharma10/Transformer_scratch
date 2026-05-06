def extract(pcmiter, samplerate, channels, duration = -1):
    """Given a PCM data stream, extract fingerprint data from the
    audio. Returns a byte string of fingerprint data. Raises an
    ExtractionError if fingerprinting fails.
    """
    extractor = _fplib.Extractor(samplerate, channels, duration)

    # Get first block.
    try:
        next_block = next(pcmiter)
    except StopIteration:
        raise ExtractionError()

    # Get and process subsequent blocks.
    while True:
        # Shift over blocks.
        cur_block = next_block
        try:
            next_block = next(pcmiter)
        except StopIteration:
            next_block = None
        done = next_block is None

        # Process the block.
        try:
            if extractor.process(cur_block, done):
                # Success!
                break
        except RuntimeError as exc:
            # Exception from fplib. Most likely the file is too short.
            raise ExtractionError(exc.args[0])

        # End of file but processor never became ready?
        if done:
            raise ExtractionError()

    # Get resulting fingerprint data.
    out = extractor.result()
    if out is None:
        raise ExtractionError()
    
    # Free extractor memory.
    extractor.free()
    
    return out