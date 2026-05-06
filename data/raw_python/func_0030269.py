def get_or_create_placeholder(page, placeholder_slot, delete_existing=False):
    """
    Get or create a placeholder on the given page.
    Optional: Delete existing placeholder.
    """
    placeholder, created = page.placeholders.get_or_create(
        slot=placeholder_slot)
    if created:
        log.debug("Create placeholder %r for page %r", placeholder_slot,
                  page.get_title())
    else:
        log.debug("Use existing placeholder %r for page %r", placeholder_slot,
                  page.get_title())

    if delete_existing:
        queryset = CMSPlugin.objects.all().filter(placeholder=placeholder)
        log.info("Delete %i CMSPlugins on placeholder %s...", queryset.count(),
                 placeholder)
        queryset.delete()

    return placeholder, created