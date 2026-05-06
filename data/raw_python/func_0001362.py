def wf(raw_str: str,
       flush: bool = True,
       prevent_completion_polluting: bool = True,
       stream: t.TextIO = sys.stdout):
    """
    Writes a given ``raw_str`` into a ``stream``. Ignores output if ``prevent_completion_polluting`` is set and there's
    no extra ``sys.argv`` arguments present (a bash completion issue).

    :param raw_str: a raw string to print
    :param flush: execute ``flush()``
    :param prevent_completion_polluting: don't write anything if ``len(sys.argv) <= 1``
    :param stream: ``sys.stdout`` by default
    :return: None
    """
    if prevent_completion_polluting and len(sys.argv) <= 1:
        return

    stream.write(raw_str)
    flush and hasattr(stream, 'flush') and stream.flush()