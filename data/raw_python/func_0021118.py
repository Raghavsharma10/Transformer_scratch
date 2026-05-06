def set_slug(apps, schema_editor):
    """
    Create a slug for each Creator already in the DB.
    """
    Creator = apps.get_model('spectator_core', 'Creator')

    for c in Creator.objects.all():
        c.slug = generate_slug(c.pk)
        c.save(update_fields=['slug'])