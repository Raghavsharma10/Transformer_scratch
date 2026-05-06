def local():
    """Load local requirements file."""
    logger.info("Loading requirements from local file.")
    with open(REQUIREMENTS_FILE, 'r') as f:
        requirements = parse(f)
        for r in requirements:
            logger.debug("Creating new package: %r", r)
            create_package_version(r)