def calendar(request, year=None, month=None, template="gnotty/calendar.html"):
    """
    Show calendar months for the given year/month.
    """

    try:
        year = int(year)
    except TypeError:
        year = datetime.now().year
    lookup = {"message_time__year": year}
    if month:
        lookup["message_time__month"] = month
    if hide_joins_and_leaves(request):
        lookup["join_or_leave"] = False

    messages = IRCMessage.objects.filter(**lookup)
    try:
        dates = messages.datetimes("message_time", "day")
    except AttributeError:
        dates = messages.dates("message_time", "day")
    days = [d.date() for d in dates]
    months = []

    if days:
        min_date, max_date = days[0], days[-1]
        days = set(days)
        calendar = Calendar(SUNDAY)
        for m in range(1, 13) if not month else [int(month)]:
            lt_max = m <= max_date.month or year < max_date.year
            gt_min = m >= min_date.month or year > min_date.year
            if lt_max and gt_min:
                weeks = calendar.monthdatescalendar(year, m)
                for w, week in enumerate(weeks):
                    for d, day in enumerate(week):
                        weeks[w][d] = {
                            "date": day,
                            "in_month": day.month == m,
                            "has_messages": day in days,
                        }
                months.append({"month": date(year, m, 1), "weeks": weeks})

    context = dict(settings)
    context["months"] = months
    return render(request, template, context)