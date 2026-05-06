def generate_schedule(today=None):
    """Helper function which creates an ordered list of schedule days"""
    # We create a list of slots and schedule items
    schedule_days = {}
    seen_items = {}
    for slot in Slot.objects.all().order_by('end_time', 'start_time', 'day'):
        day = slot.get_day()
        if today and day != today:
            # Restrict ourselves to only today
            continue
        schedule_day = schedule_days.get(day)
        if schedule_day is None:
            schedule_day = schedule_days[day] = ScheduleDay(day)
        row = make_schedule_row(schedule_day, slot, seen_items)
        schedule_day.rows.append(row)
    return sorted(schedule_days.values(), key=lambda x: x.day.date)