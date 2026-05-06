def opening_hours(location=None, concise=False):
    """
    Creates a rendered listing of hours.
    """
    template_name = 'openinghours/opening_hours_list.html'
    days = []  # [{'hours': '9:00am to 5:00pm', 'name': u'Monday'}, {'hours...

    # Without `location`, choose the first company.
    if location:
        ohrs = OpeningHours.objects.filter(company=location)
    else:
        try:
            Location = utils.get_premises_model()
            ohrs = Location.objects.first().openinghours_set.all()
        except AttributeError:
            raise Exception("You must define some opening hours"
                            " to use the opening hours tags.")

    ohrs.order_by('weekday', 'from_hour')

    for o in ohrs:
        days.append({
            'day_number': o.weekday,
            'name': o.get_weekday_display(),
            'from_hour': o.from_hour,
            'to_hour': o.to_hour,
            'hours': '%s%s to %s%s' % (
                o.from_hour.strftime('%I:%M').lstrip('0'),
                o.from_hour.strftime('%p').lower(),
                o.to_hour.strftime('%I:%M').lstrip('0'),
                o.to_hour.strftime('%p').lower()
            )
        })

    open_days = [o.weekday for o in ohrs]
    for day_number, day_name in WEEKDAYS:
        if day_number not in open_days:
            days.append({
                'day_number': day_number,
                'name': day_name,
                'hours': 'Closed'
            })
    days = sorted(days, key=lambda k: k['day_number'])

    if concise:
        # [{'hours': '9:00am to 5:00pm', 'day_names': u'Monday to Friday'},
        #  {'hours':...
        template_name = 'openinghours/opening_hours_list_concise.html'
        concise_days = []
        current_set = {}
        for day in days:
            if 'hours' not in current_set.keys():
                current_set = {'day_names': [day['name']],
                               'hours': day['hours']}
            elif day['hours'] != current_set['hours']:
                concise_days.append(current_set)
                current_set = {'day_names': [day['name']],
                               'hours': day['hours']}
            else:
                current_set['day_names'].append(day['name'])
        concise_days.append(current_set)

        for day_set in concise_days:
            if len(day_set['day_names']) > 2:
                day_set['day_names'] = '%s to %s' % (day_set['day_names'][0],
                                                     day_set['day_names'][-1])
            elif len(day_set['day_names']) > 1:
                day_set['day_names'] = '%s and %s' % (day_set['day_names'][0],
                                                      day_set['day_names'][-1])
            else:
                day_set['day_names'] = '%s' % day_set['day_names'][0]

        days = concise_days

    template = get_template(template_name)
    return template.render({'days': days})