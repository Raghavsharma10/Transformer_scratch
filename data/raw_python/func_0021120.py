def forwards(apps, schema_editor):
    """
    Set the venue_name field of all Events that have a Venue.
    """
    Event = apps.get_model('spectator_events', 'Event')

    for event in Event.objects.all():
        if event.venue is not None:
            event.venue_name = event.venue.name
            event.save()