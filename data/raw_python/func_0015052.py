def print_filters():
    """Prints all filters available with their description."""
    for filter_name in VALID_FILTERS:
        filter_func = getattr(filters, 'filter_{0}'.format(filter_name))
        description = filter_func.__doc__
        if description:
            description = re.sub(r'\n\s+', ' ', description)
            description.strip()

        print('{0}: {1}\n'.format(filter_name, description))