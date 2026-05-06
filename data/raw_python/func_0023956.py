def find_duplicate_schedule_items(all_items):
    """Find talks / pages assigned to mulitple schedule items"""
    duplicates = []
    seen_talks = {}
    for item in all_items:
        if item.talk and item.talk in seen_talks:
            duplicates.append(item)
            if seen_talks[item.talk] not in duplicates:
                duplicates.append(seen_talks[item.talk])
        else:
            seen_talks[item.talk] = item
        # We currently allow duplicate pages for cases were we need disjoint
        # schedule items, like multiple open space sessions on different
        # days and similar cases. This may be revisited later
    return duplicates