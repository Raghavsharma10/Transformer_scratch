def hfoslog(*what, **kwargs):
    """Logs all *what arguments.

    :param *what: Loggable objects (i.e. they have a string representation)
    :param lvl: Debug message level
    :param exc: Switch to better handle exceptions, use if logging in an
                except clause
    :param emitter: Optional log source, where this can't be determined
                    automatically
    :param sourceloc: Give specific source code location hints, used internally
    """

    # Count all messages (missing numbers give a hint at too high log level)
    global count
    global verbosity

    count += 1

    lvl = kwargs.get('lvl', info)

    if lvl < verbosity['global']:
        return

    emitter = kwargs.get('emitter', 'UNKNOWN')
    traceback = kwargs.get('tb', False)
    frame_ref = kwargs.get('frame_ref', 0)

    output = None

    timestamp = time.time()
    runtime = timestamp - start

    callee = None

    exception = kwargs.get('exc', False)

    if exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()  # NOQA

    if verbosity['global'] <= debug or traceback:
        # Automatically log the current function details.

        if 'sourceloc' not in kwargs:
            frame = kwargs.get('frame', frame_ref)

            # Get the previous frame in the stack, otherwise it would
            # be this function
            current_frame = inspect.currentframe()
            while frame > 0:
                frame -= 1
                current_frame = current_frame.f_back

            func = current_frame.f_code
            # Dump the message + the name of this function to the log.

            if exception:
                line_no = exc_tb.tb_lineno
                if lvl <= error:
                    lvl = error
            else:
                line_no = func.co_firstlineno

            callee = "[%.10s@%s:%i]" % (
                func.co_name,
                func.co_filename,
                line_no
            )
        else:
            callee = kwargs['sourceloc']

    now = time.asctime()

    msg = "[%s] : %5s : %.5f : %3i : [%5s]" % (now,
                                               lvldata[lvl][0],
                                               runtime,
                                               count,
                                               emitter)

    content = ""

    if callee:
        if not uncut and lvl > 10:
            msg += "%-60s" % callee
        else:
            msg += "%s" % callee

    for thing in what:
        content += " "
        if kwargs.get('pretty', False):
            content += pprint.pformat(thing)
        else:
            content += str(thing)

    msg += content

    if exception:
        msg += "\n" + "".join(format_exception(exc_type, exc_obj, exc_tb))

    if is_muted(msg):
        return

    if not uncut and lvl > 10 and len(msg) > 1000:
        msg = msg[:1000]

    if lvl >= verbosity['file']:
        try:
            f = open(logfile, "a")
            f.write(msg + '\n')
            f.flush()
            f.close()
        except IOError:
            print("Can't open logfile %s for writing!" % logfile)
            # sys.exit(23)

    if is_marked(msg):
        lvl = hilight

    if lvl >= verbosity['console']:
        output = str(msg)
        if six.PY3 and color:
            output = lvldata[lvl][1] + output + terminator
        try:
            print(output)
        except UnicodeEncodeError as e:
            print(output.encode("utf-8"))
            hfoslog("Bad encoding encountered on previous message:", e,
                    lvl=error)
        except BlockingIOError:
            hfoslog("Too long log line encountered:", output[:20], lvl=warn)

    if live:
        item = [now, lvl, runtime, count, emitter, str(content)]
        LiveLog.append(item)