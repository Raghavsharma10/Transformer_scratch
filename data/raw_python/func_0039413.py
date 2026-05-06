def parse_journal(journal):
    """Parses the USN Journal content removing duplicates
    and corrupted records.

    """
    events = [e for e in journal if not isinstance(e, CorruptedUsnRecord)]
    keyfunc = lambda e: str(e.file_reference_number) + e.file_name + e.timestamp
    event_groups = (tuple(g) for k, g in groupby(events, key=keyfunc))

    if len(events) < len(list(journal)):
        LOGGER.debug(
            "Corrupted records in UsnJrnl, some events might be missing.")

    return [journal_event(g) for g in event_groups]