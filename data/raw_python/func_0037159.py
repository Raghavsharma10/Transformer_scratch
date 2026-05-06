def create_package_version(requirement):
    """Create a new PackageVersion from a requirement. Handles errors."""
    try:
        PackageVersion(requirement=requirement).save()
        logger.info("Package '%s' added.", requirement.name)  # noqa
    except IntegrityError:
        logger.info("Package '%s' already exists.", requirement.name)