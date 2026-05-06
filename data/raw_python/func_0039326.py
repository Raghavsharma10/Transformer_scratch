def reversetext(contenttoreverse, reconvert=True):
    """
    Reverse any content

    :type contenttoreverse: string
    :param contenttoreverse: The content to be reversed

    :type reeval: boolean
    :param reeval: Wether or not to reconvert the object back into it's initial state. Default is "True".
    """

    # If reconvert is specified
    if reconvert is True:
        # Return the evalated form
        return eval(
            str(type(contenttoreverse)).split("'")[1] + "('" +
            str(contenttoreverse)[::-1] + "')")

    # Return the raw version
    return contenttoreverse[::-1]