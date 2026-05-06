def load_template():
    """Bail out if template is not found.
    """
    cloudformation, found = load_cloudformation_template()
    if not found:
        print(colored.red('could not load cloudformation.py, bailing out...'))
        sys.exit(1)
    return cloudformation