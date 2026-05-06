def set_slug(apps, schema_editor, class_name):
    """
    Create a slug for each Work already in the DB.
    """
    Cls = apps.get_model('spectator_events', class_name)

    for obj in Cls.objects.all():
        obj.slug = generate_slug(obj.pk)
        obj.save(update_fields=['slug'])