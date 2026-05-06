def messages(request, year=None, month=None, day=None,
             template="gnotty/messages.html"):
    """
    Show messages for the given query or day.
    """

    query = request.REQUEST.get("q")
    prev_url, next_url = None, None
    messages = IRCMessage.objects.all()
    if hide_joins_and_leaves(request):
        messages = messages.filter(join_or_leave=False)
    if query:
        search = Q(message__icontains=query) | Q(nickname__icontains=query)
        messages = messages.filter(search).order_by("-message_time")
    elif year and month and day:
        messages = messages.filter(message_time__year=year,
                                   message_time__month=month,
                                   message_time__day=day)
        day_delta = timedelta(days=1)
        this_date = date(int(year), int(month), int(day))
        prev_date = this_date - day_delta
        next_date = this_date + day_delta
        prev_url = reverse("gnotty_day", args=prev_date.timetuple()[:3])
        next_url = reverse("gnotty_day", args=next_date.timetuple()[:3])
    else:
        return redirect("gnotty_year", year=datetime.now().year)

    context = dict(settings)
    context["messages"] = messages
    context["prev_url"] = prev_url
    context["next_url"] = next_url
    return render(request, template, context)