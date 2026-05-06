def execEmbCode(SCOPE, NAME, VAL, TEAL, codeStr):
    """ .cfgspc embedded code execution is done here, in a relatively confined
        space.  The variables available to the code to be executed are:
              SCOPE, NAME, VAL, PARENT, TEAL
        The code string itself is expected to set a var named OUT
    """
    # This was all we needed in Python 2.x
#   OUT = None
#   exec codeStr
#   return OUT

    # In Python 3 (& 2.x) be more explicit:  http://bugs.python.org/issue4831
    PARENT = None
    if TEAL:
        PARENT = TEAL.top
    OUT = None
    ldict = locals() # will have OUT in it
    exec(codeStr, globals(), ldict)
    return ldict['OUT']