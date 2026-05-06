def translate_exception(exc_info, initial_skip=0):
    """If passed an exc_info it will automatically rewrite the exceptions
    all the way down to the correct line numbers and frames.
    """
    tb = exc_info[2]
    frames = []

    # skip some internal frames if wanted
    for x in range(initial_skip):
        if tb is not None:
            tb = tb.tb_next
    initial_tb = tb

    while tb is not None:
        # skip frames decorated with @internalcode.  These are internal
        # calls we can't avoid and that are useless in template debugging
        # output.
        if tb.tb_frame.f_code in internal_code:
            tb = tb.tb_next
            continue

        # save a reference to the next frame if we override the current
        # one with a faked one.
        next = tb.tb_next

        # fake template exceptions
        template = tb.tb_frame.f_globals.get('__TK_template_info__')
        if template is not None:
            lineno = template.get_corresponding_lineno(tb.tb_lineno)
            tb = fake_exc_info(exc_info[:2] + (tb,), template.filename,
                               lineno)[2]

        frames.append(make_frame_proxy(tb))
        tb = next

    # if we don't have any exceptions in the frames left, we have to reraise it unchanged. XXX: can we backup here?  when could this happen?
    if not frames:
        reraise(exc_info[0], exc_info[1], exc_info[2])

    return ProcessedTraceback(exc_info[0], exc_info[1], frames)