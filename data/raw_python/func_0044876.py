def format_frameinfo(fi):
    """
    Takes a frameinfo object (from the inspect module)

    returns a properly formated string
    """
    s1 = "{0}:{1}".format(fi.filename, fi.lineno)
    s2 = "function:{0},    code_context:".format(fi.function)
    if fi.code_context:
        s3 = fi.code_context[0]
    else:
        s3 = "<no code context available>"

    return "\n".join([s1, s2, s3])