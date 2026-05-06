def connect(slug, config_loader):
    """
    Ensure .cs50.yaml and tool key exists, raises Error otherwise
    Check that all required files as per .cs50.yaml are present
    Returns tool specific portion of .cs50.yaml
    """
    with ProgressBar(_("Connecting")):
        # Parse slug
        slug = Slug(slug)

        # Get .cs50.yaml
        try:
            config = config_loader.load(_get_content(slug.org, slug.repo,
                                              slug.branch, slug.problem / ".cs50.yaml"))
        except InvalidConfigError:
            raise InvalidSlugError(_("Invalid slug for {}. Did you mean something else?").format(config_loader.tool))

        print("WTF!!!!", config)

        if not config:
            raise InvalidSlugError(_("Invalid slug for {}. Did you mean something else?").format(config_loader.tool))

        # If config of tool is just a truthy value, config should be empty
        if not isinstance(config, dict):
            config = {}

        org = config.get("org", config_loader.tool)
        included, excluded = files(config.get("files"))

        # Check that at least 1 file is staged
        if not included:
            raise Error(_("No files in this directory are expected for submission."))

        return org, (included, excluded)