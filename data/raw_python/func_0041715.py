def _component_of(name):
    """Get the root package or module of the passed module.
    """

    # Get the registered package this model belongs to.
    segments = name.split('.')
    while segments:
        # Is this name a registered package?
        test = '.'.join(segments)
        if test in settings.get('COMPONENTS', []):
            # This is the component we are in.
            return test

        # Remove the right-most segment.
        segments.pop()

    if not segments and '.models' in name:
        # No package was found to be registered; attempt to guess the
        # right package name; strip all occurrances of '.models' from the
        # pacakge name.
        return _component_of(name.replace('.models', ''))