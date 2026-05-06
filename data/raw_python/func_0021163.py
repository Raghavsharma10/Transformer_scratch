def forwards(apps, schema_editor):
    """
    Change Events with kind 'movie' to 'cinema'
    and Events with kind 'play' to 'theatre'.

    Purely for more consistency.
    """
    Event = apps.get_model('spectator_events', 'Event')

    for ev in Event.objects.filter(kind='movie'):
        ev.kind = 'cinema'
        ev.save()

    for ev in Event.objects.filter(kind='play'):
        ev.kind = 'theatre'
        ev.save()