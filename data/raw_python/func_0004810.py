def _check_environ(variable, value):
    """check if a variable is present in the environmental variables"""
    if is_not_none(value):
        return value
    else:
        value = os.environ.get(variable)
        if is_none(value):
            stop(''.join([variable,
                          """ not supplied and no entry in environmental
                           variables"""]))
        else:
            return value