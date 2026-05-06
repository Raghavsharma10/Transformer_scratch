def _validate_domains(domains):
        """Check that all domains specified in the settings was provided in the options."""
        missing = set(settings.TRANSLATION_DOMAINS.keys()) - set(domains.keys())
        if missing:
            print('The following domains have been set in the configuration, '
                  'but their sources were not provided, use the `--source` '
                  'option to specify their sources: {domains}'.format(domains=', '.join(missing)))
            sys.exit(1)