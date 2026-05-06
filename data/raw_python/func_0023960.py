def validate_schedule():
    """Helper routine to report issues with the schedule"""
    all_items = prefetch_schedule_items()
    errors = []
    for validator, _type, msg in SCHEDULE_ITEM_VALIDATORS:
        if validator(all_items):
            errors.append(msg)

    all_slots = prefetch_slots()
    for validator, _type, msg in SLOT_VALIDATORS:
        if validator(all_slots):
            errors.append(msg)
    return errors