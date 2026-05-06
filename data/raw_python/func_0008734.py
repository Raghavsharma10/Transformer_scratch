def _extract_traceback(text):
    """Receive a list of strings representing the input from stdin and return
    the restructured backtrace.

    This iterates over the output and once it identifies a hopefully genuine
    identifier, it will start parsing output.
    In the case the input includes a reraise (a Python 3 case), the primary
    traceback isn't handled, only the reraise.

    Each of the traceback lines are then handled two lines at a time for each
    stack object.

    Note that all parts of each stack object are stripped from newlines and
    spaces to keep the output clean.
    """
    capture = False
    entries = []
    all_else = []
    ignore_trace = False

    # In python 3, a traceback may includes output from a reraise.
    # e.g, an exception is captured and reraised with another exception.
    # This marks that we should ignore
    if text.count(TRACEBACK_IDENTIFIER) == 2:
        ignore_trace = True

    for index, line in enumerate(text):
        if TRACEBACK_IDENTIFIER in line:
            if ignore_trace:
                ignore_trace = False
                continue
            capture = True
        # We're not capturing and making sure we only read lines
        # with spaces since, after the initial identifier, all traceback lines
        # contain a prefix spacing.
        elif capture and line.startswith(' '):
            if index % 2 == 0:
                # Line containing a file, line and module.
                line = line.strip().strip('\n')
                next_line = text[index + 1].strip('\n')
                entries.append(line + ', ' + next_line)
        elif capture:
            # Line containing the module call.
            entries.append(line)
            break
        else:
            # Add everything else after the traceback.
            all_else.append(line)

    traceback_entries = []

    # Build the traceback structure later passed for formatting.
    for index, line in enumerate(entries[:-2]):
        # TODO: This should be done in a _parse_entry function
        element = line.split(',')
        element[0] = element[0].strip().lstrip('File').strip(' "')
        element[1] = element[1].strip().lstrip('line').strip()
        element[2] = element[2].strip().lstrip('in').strip()
        traceback_entries.append(tuple(element))
    return traceback_entries, all_else