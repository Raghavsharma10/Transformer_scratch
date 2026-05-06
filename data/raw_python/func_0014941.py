def makePrintReturner(pre="", post="" ,out=None):
    r"""Creates functions that print out their argument, (between optional
    `pre` and `post` strings) and return it unmodified. This is usefull for
    debugging e.g. parts of expressions, without having to modify the behavior
    of the program.

    Example:

    >>> makePrintReturner(pre="The value is:", post="[returning]")(3)
    The value is: 3 [returning]
    3
    >>>
    """
    def printReturner(arg):
        myArgs = [pre, arg, post]
        prin(*myArgs, **{'out':out})
        return arg
    return printReturner