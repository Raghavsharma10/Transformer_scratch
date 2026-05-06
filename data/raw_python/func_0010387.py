def is_special_atom(cron_atom, span):
    """
    Returns a boolean indicating whether or not the string can be parsed by
    parse_atom to produce a static set. In the process of examining the
    string, the syntax of any special character uses is also checked.
    """
    for special_char in ('%', '#', 'L', 'W'):
        if special_char not in cron_atom:
            continue

        if special_char == '#':
            if span != DAYS_OF_WEEK:
                raise ValueError("\"#\" invalid where used.")
            elif not VALIDATE_POUND.match(cron_atom):
                raise ValueError("\"#\" syntax incorrect.")
        elif special_char == "W":
            if span != DAYS_OF_MONTH:
                raise ValueError("\"W\" syntax incorrect.")
            elif not(VALIDATE_W.match(cron_atom) and int(cron_atom[:-1]) > 0):
                raise ValueError("Invalid use of \"W\".")
        elif special_char == "L":
            if span not in L_FIELDS:
                raise ValueError("\"L\" invalid where used.")
            elif span == DAYS_OF_MONTH:
                if cron_atom != "L":
                    raise ValueError("\"L\" must be alone in days of month.")
            elif span == DAYS_OF_WEEK:
                if not VALIDATE_L_IN_DOW.match(cron_atom):
                    raise ValueError("\"L\" syntax incorrect.")
        elif special_char == "%":
            if not(cron_atom[1:].isdigit() and int(cron_atom[1:]) > 1):
                raise ValueError("\"%\" syntax incorrect.")
        return True
    else:
        return False