def splitValues(textStr):
    """Splits a comma-separated number sequence into a list (of floats).
    """
    vals = textStr.split(",")
    nums = []
    for v in vals:
        nums.append(float(v))
    return nums