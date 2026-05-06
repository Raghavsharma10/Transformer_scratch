def is_lossy(label):
    """Check Label file for the compression type. """
    val = getkey(from_=label, keyword='INST_CMPRS_TYPE').decode().strip()
    if val == 'LOSSY':
        return True
    else:
        return False