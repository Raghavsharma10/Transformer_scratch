def check_pypi(modeladmin, request, queryset):
    """Update latest package info from PyPI."""
    for p in queryset:
        if p.is_editable:
            logger.debug("Ignoring version update '%s' is editable", p.package_name)
        else:
            p.update_from_pypi()