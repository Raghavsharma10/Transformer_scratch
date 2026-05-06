def checkrc(rc, *args):
    """Check if ``rc`` (return code) meets requirements.

    Check if ``rc`` is 0 or is in ``args`` list that contains
    acceptable return codes.
    Last argument of ``args`` can optionally be error message that
    is printed if ``rc`` doesn't meet requirements.

    Output is JSON of the form:

        {"proc.rc": <rc>,
         "proc.error": "<error_msg>"},

    where "proc.error" entry is omitted if empty.

    """
    try:
        rc = int(rc)
    except (TypeError, ValueError):
        return error("Invalid return code: '{}'.".format(rc))

    acceptable_rcs = []
    error_msg = ""

    if len(args):
        for code in args[:-1]:
            try:
                acceptable_rcs.append(int(code))
            except (TypeError, ValueError):
                return error("Invalid return code: '{}'.".format(code))

        try:
            acceptable_rcs.append(int(args[-1]))
        except (TypeError, ValueError):
            error_msg = args[-1]

    if rc in acceptable_rcs:
        rc = 0

    ret = {'proc.rc': rc}
    if rc and error_msg:
        ret['proc.error'] = error_msg

    return json.dumps(ret)