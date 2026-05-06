def find_clashes(all_items):
    """Find schedule items which clash (common slot and venue)"""
    clashes = {}
    seen_venue_slots = {}
    for item in all_items:
        for slot in item.slots.all():
            pos = (item.venue, slot)
            if pos in seen_venue_slots:
                if seen_venue_slots[pos] not in clashes:
                    clashes[pos] = [seen_venue_slots[pos]]
                clashes[pos].append(item)
            else:
                seen_venue_slots[pos] = item
    # We return a list, to match other validators
    return clashes.items()