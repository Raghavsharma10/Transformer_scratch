def update_schedule_items(*args, **kw):
    """We save all the schedule items associated with this slot, so
       the last_update time is updated to reflect any changes to the
       timing of the slots"""
    slot = kw.pop('instance', None)
    if not slot:
        return
    for item in slot.scheduleitem_set.all():
        item.save(update_fields=['last_updated'])
    # We also need to update the next slot, in case we changed it's
    # times as well
    next_slot = slot.slot_set.all()
    if next_slot.count():
        # From the way we structure the slot tree, we know that
        # there's only 1 next slot that could have changed.
        for item in next_slot[0].scheduleitem_set.all():
            item.save(update_fields=['last_updated'])