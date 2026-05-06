def journal_event(events):
    """Group multiple events into a single one."""
    reasons = set(chain.from_iterable(e.reasons for e in events))
    attributes = set(chain.from_iterable(e.file_attributes for e in events))

    return JrnlEvent(events[0].file_reference_number,
                     events[0].parent_file_reference_number,
                     events[0].file_name,
                     events[0].timestamp,
                     list(reasons), list(attributes))