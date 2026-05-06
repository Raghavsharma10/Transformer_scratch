def remote():
    """Update package info from PyPI."""
    logger.info("Fetching latest data from PyPI.")
    results = defaultdict(list)
    packages = PackageVersion.objects.exclude(is_editable=True)
    for pv in packages:
        pv.update_from_pypi()
        results[pv.diff_status].append(pv)
        logger.debug("Updated package from PyPI: %r", pv)
    results['refreshed_at'] = tz_now()
    return results