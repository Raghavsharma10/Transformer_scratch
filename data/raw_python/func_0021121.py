def forwards(apps, schema_editor):
    """
    Migrate all 'exhibition' Events to the new 'museum' Event kind.
    """
    Event = apps.get_model('spectator_events', 'Event')

    for ev in Event.objects.filter(kind='exhibition'):
        ev.kind = 'museum'
        ev.save()