def fsliceafter(astr, sub):
    """Return the slice after at sub in string astr"""
    findex = astr.find(sub)
    return astr[findex + len(sub):]