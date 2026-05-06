def find_non_contiguous(all_items):
    """Find any items that have slots that aren't contiguous"""
    non_contiguous = []
    for item in all_items:
        if item.slots.count() < 2:
            # No point in checking
            continue
        last_slot = None
        for slot in item.slots.all().order_by('end_time'):
            if last_slot:
                if last_slot.end_time != slot.get_start_time():
                    non_contiguous.append(item)
                    break
            last_slot = slot
    return non_contiguous