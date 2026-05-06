def synth_hangul(string):
    """Convert jamo characters in a string into hcj as much as possible."""
    raise NotImplementedError
    return ''.join([''.join(''.join(jamo_to_hcj(_)) for _ in string)])