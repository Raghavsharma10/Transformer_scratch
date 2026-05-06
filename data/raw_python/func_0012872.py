def removeblanklines(astr):
    """remove the blank lines in astr"""
    lines = astr.splitlines()
    lines = [line for line in lines if line.strip() != ""]
    return "\n".join(lines)