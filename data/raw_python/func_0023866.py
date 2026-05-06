def make_schedule_row(schedule_day, slot, seen_items):
    """Create a row for the schedule table."""
    row = ScheduleRow(schedule_day, slot)
    skip = {}
    expanding = {}
    all_items = list(slot.scheduleitem_set
                     .select_related('talk', 'page', 'venue')
                     .all())

    for item in all_items:
        if item in seen_items:
            # Inc rowspan
            seen_items[item]['rowspan'] += 1
            # Note that we need to skip this during colspan checks
            skip[item.venue] = seen_items[item]
            continue
        scheditem = {'item': item, 'rowspan': 1, 'colspan': 1}
        row.items[item.venue] = scheditem
        seen_items[item] = scheditem
        if item.expand:
            expanding[item.venue] = []

    empty = []
    expanding_right = None
    skipping = 0
    skip_item = None
    for venue in schedule_day.venues:
        if venue in skip:
            # We need to skip all the venues this item spans over
            skipping = 1
            skip_item = skip[venue]
            continue
        if venue in expanding:
            item = row.items[venue]
            for empty_venue in empty:
                row.items.pop(empty_venue)
                item['colspan'] += 1
            empty = []
            expanding_right = item
        elif venue in row.items:
            empty = []
            expanding_right = None
        elif expanding_right:
            expanding_right['colspan'] += 1
        elif skipping > 0 and skipping < skip_item['colspan']:
            skipping += 1
        else:
            skipping = 0
            empty.append(venue)
            row.items[venue] = {'item': None, 'rowspan': 1, 'colspan': 1}

    return row