def find_invalid_venues(all_items):
    """Find venues assigned slots that aren't on the allowed list
       of days."""
    venues = {}
    for item in all_items:
        valid = False
        item_days = list(item.venue.days.all())
        for slot in item.slots.all():
            for day in item_days:
                if day == slot.get_day():
                    valid = True
                    break
        if not valid:
            venues.setdefault(item.venue, [])
            venues[item.venue].append(item)
    return venues.items()