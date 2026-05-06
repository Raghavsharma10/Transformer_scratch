def stackToList(stack):
    """
    Convert a chain of traceback or frame objects into a list of frames.
    """
    if isinstance(stack, types.TracebackType):
        while stack.tb_next:
            stack = stack.tb_next
        stack = stack.tb_frame
    out = []
    while stack:
        out.append(stack)
        stack = stack.f_back
    return out