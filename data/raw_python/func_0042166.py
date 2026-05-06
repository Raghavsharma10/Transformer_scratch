def _wait_for_exit(please_stop):
    """
    /dev/null PIPED TO sys.stdin SPEWS INFINITE LINES, DO NOT POLL AS OFTEN
    """
    try:
        import msvcrt
        _wait_for_exit_on_windows(please_stop)
    except:
        pass

    cr_count = 0  # COUNT NUMBER OF BLANK LINES

    while not please_stop:
        # DEBUG and Log.note("inside wait-for-shutdown loop")
        if cr_count > 30:
            (Till(seconds=3) | please_stop).wait()
        try:
            line = sys.stdin.readline()
        except Exception as e:
            Except.wrap(e)
            if "Bad file descriptor" in e:
                _wait_for_interrupt(please_stop)
                break

        # DEBUG and Log.note("read line {{line|quote}}, count={{count}}", line=line, count=cr_count)
        if line == "":
            cr_count += 1
        else:
            cr_count = -1000000  # NOT /dev/null

        if line.strip() == "exit":
            Log.alert("'exit' Detected!  Stopping...")
            return