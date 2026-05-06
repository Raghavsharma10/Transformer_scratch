def getExceptionClass(errorCode):
    """
    Converts the specified error code into the corresponding class object.
    Raises a KeyError if the errorCode is not found.
    """
    classMap = {}
    for name, class_ in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(class_) and issubclass(class_, BaseServerException):
            classMap[class_.getErrorCode()] = class_
    return classMap[errorCode]