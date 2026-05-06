def parse_log_entry(text):
    """This function does all real job on log line parsing.
    it setup two cases for restart parsing if a line
    with wrong format was found.

    Restarts:
    - use_value: just retuns an object it was passed. This can
      be any value.
    - reparse: calls `parse_log_entry` again with other text value.
      Beware, this call can lead to infinite recursion.
    """
    text = text.strip()

    if well_formed_log_entry_p(text):
        return LogEntry(text)
    else:
        def use_value(obj):
            return obj
        def reparse(text):
            return parse_log_entry(text)

        with restarts(use_value,
                      reparse) as call:
            return call(signal, MalformedLogEntryError(text))