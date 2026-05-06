def get(self, request):
        """Create a iCal file from the schedule"""
        # Heavily inspired by https://djangosnippets.org/snippets/2223/ and
        # the icalendar documentation
        calendar = Calendar()
        site = get_current_site(request)
        calendar.add('prodid', '-//%s Schedule//%s//' % (site.name, site.domain))
        calendar.add('version', '2.0')

        # Since we don't need to format anything here, we can just use a list
        # of schedule items
        for item in ScheduleItem.objects.all():
            sched_event = Event()
            sched_event.add('dtstamp', item.last_updated)
            sched_event.add('summary', item.get_title())
            sched_event.add('location', item.venue.name)
            sched_event.add('dtstart', item.get_start_datetime())
            sched_event.add('duration', datetime.timedelta(minutes=item.get_duration_minutes()))
            sched_event.add('class', 'PUBLIC')
            sched_event.add('uid', '%s@%s' % (item.pk, site.domain))
            calendar.add_component(sched_event)
        response = HttpResponse(calendar.to_ical(), content_type="text/calendar")
        response['Content-Disposition'] = 'attachment; filename=schedule.ics'
        return response