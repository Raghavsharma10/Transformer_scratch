def find_overlapping_slots(all_slots):
    """Find any slots that overlap"""
    overlaps = set([])
    for slot in all_slots:
        # Because slots are ordered, we can be more efficient than this
        # N^2 loop, but this is simple and, since the number of slots
        # should be low, this should be "fast enough"
        start = slot.get_start_time()
        end = slot.end_time
        for other_slot in all_slots:
            if other_slot.pk == slot.pk:
                continue
            if other_slot.get_day() != slot.get_day():
                # different days, can't overlap
                continue
            # Overlap if the start_time or end_time is bounded by our times
            # start_time <= other.start_time < end_time
            # or
            # start_time < other.end_time <= end_time
            other_start = other_slot.get_start_time()
            other_end = other_slot.end_time
            if start <= other_start and other_start < end:
                overlaps.add(slot)
                overlaps.add(other_slot)
            elif start < other_end and other_end <= end:
                overlaps.add(slot)
                overlaps.add(other_slot)
    return overlaps