def check_schedule():
    """Helper routine to easily test if the schedule is valid"""
    all_items = prefetch_schedule_items()
    for validator, _type, _msg in SCHEDULE_ITEM_VALIDATORS:
        if validator(all_items):
            return False

    all_slots = prefetch_slots()
    for validator, _type, _msg in SLOT_VALIDATORS:
        if validator(all_slots):
            return False
    return True