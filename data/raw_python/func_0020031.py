def site_cleanup(sender, action, instance, **kwargs):
    """
    Make sure there is only a single preferences object per site.
    So remove sites from pre-existing preferences objects.
    """
    if action == 'post_add':
        if isinstance(instance, Preferences) \
            and hasattr(instance.__class__, 'objects'):
            site_conflicts = instance.__class__.objects.filter(
                sites__in=instance.sites.all()
            ).only('id').distinct()

            for conflict in site_conflicts:
                if conflict.id != instance.id:
                    for site in instance.sites.all():
                        conflict.sites.remove(site)