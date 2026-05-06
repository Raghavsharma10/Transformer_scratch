def fmt(msg, *args, **kw):
    # type: (str, *Any, **Any) -> str
    """ Generate shell color opcodes from a pretty coloring syntax. """
    global is_tty

    if len(args) or len(kw):
        msg = msg.format(*args, **kw)

    opcode_subst = '\x1b[\\1m' if is_tty else ''
    return re.sub(r'<(\d{1,2})>', opcode_subst, msg)