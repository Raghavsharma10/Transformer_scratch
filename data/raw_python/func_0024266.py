def _prepare_domain(mapping):
        """Prepare a helper dictionary for the domain to temporarily hold some information."""
        # Parse the domain-directory mapping
        try:
            domain, dir = mapping.split(':')
        except ValueError:
            print("Please provide the sources in the form of '<domain>:<directory>'")
            sys.exit(1)

        try:
            default_language = settings.TRANSLATION_DOMAINS[domain]
        except KeyError:
            print("Unknown domain {domain}, check the settings file to make sure"
                  " this domain is set in TRANSLATION_DOMAINS".format(domain=domain))
            sys.exit(1)
        # Create a temporary file to hold the `.pot` file for this domain
        handle, path = tempfile.mkstemp(prefix='zengine_i18n_', suffix='.pot')
        return (domain, {
            'default': default_language,
            'pot': path,
            'source': dir,
        })