def class_in_progress(stack=None):
    """True if currently inside a class definition, else False."""
    if stack is None:
        stack = inspect.stack()
    for frame in stack:
        statement_list = frame[4]
        if statement_list is None:
            continue
        if statement_list[0].strip().startswith('class '):
            return True
    return False