def set_slug(apps, schema_editor):
    """
    Create a slug for each Event already in the DB.
    """
    Event = apps.get_model('spectator_events', 'Event')

    for e in Event.objects.all():
        e.slug = generate_slug(e.pk)
        e.save(update_fields=['slug'])