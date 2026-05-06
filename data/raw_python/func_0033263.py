def get_frames(tback, is_breakpoint):
    """Builds a list of ErrorFrame objects from a traceback"""

    frames = []

    while tback is not None:
        if tback.tb_next is None and is_breakpoint:
            break

        filename = tback.tb_frame.f_code.co_filename
        function = tback.tb_frame.f_code.co_name
        context = tback.tb_frame.f_locals
        lineno = tback.tb_lineno - 1
        tback_id = id(tback)
        pre_context_lineno, pre_context, context_line, post_context = get_lines_from_file(filename, lineno + 1, 7)
        frames.append(ErrorFrame(tback, filename, function, lineno, context, tback_id, pre_context, context_line, post_context, pre_context_lineno))
        tback = tback.tb_next

    return frames