def forwards(apps, schema_editor):
    """
    Re-save all the Works because something earlier didn't create their slugs.
    """
    Work = apps.get_model('spectator_events', 'Work')

    for work in Work.objects.all():
        if not work.slug:
            work.slug = generate_slug(work.pk)
            work.save()