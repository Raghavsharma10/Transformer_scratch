def _string2Duration(text):
    """
    CONVERT SIMPLE <float><type> TO A DURATION OBJECT
    """
    if text == "" or text == "zero":
        return ZERO

    amount, interval = re.match(r"([\d\.]*)(.*)", text).groups()
    amount = int(amount) if amount else 1

    if MILLI_VALUES[interval] == None:
        from mo_logs import Log
        Log.error(
            "{{interval|quote}} in {{text|quote}} is not a recognized duration type (did you use the pural form by mistake?",
            interval=interval,
            text=text
        )

    output = Duration(0)
    if MONTH_VALUES[interval] == 0:
        output.milli = amount * MILLI_VALUES[interval]
    else:
        output.milli = amount * MONTH_VALUES[interval] * MILLI_VALUES.month
        output.month = amount * MONTH_VALUES[interval]

    return output