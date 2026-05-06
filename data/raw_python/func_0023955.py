def validate_items(all_items):
    """Find errors in the schedule. Check for:
         - pending / rejected talks in the schedule
         - items with both talks and pages assigned
         - items with neither talks nor pages assigned
         """
    validation = []
    for item in all_items:
        if item.talk is not None and item.page is not None:
            validation.append(item)
        elif item.talk is None and item.page is None:
            validation.append(item)
        elif item.talk and item.talk.status not in [ACCEPTED, CANCELLED]:
            validation.append(item)
    return validation