def synchronizeLayout(primary, secondary, surface_size):
    """Synchronizes given layouts by normalizing height by using
    max height of given layouts to avoid transistion dirty effects.

    :param primary: Primary layout used.
    :param secondary: Secondary layout used.
    :param surface_size: Target surface size on which layout will be displayed.
    """
    primary.configure_bound(surface_size)
    secondary.configure_bound(surface_size)
    # Check for key size.
    if (primary.key_size < secondary.key_size):
        logging.warning('Normalizing key size from secondary to primary')
        secondary.key_size = primary.key_size
    elif (primary.key_size > secondary.key_size):
        logging.warning('Normalizing key size from primary to secondary')
        primary.key_size = secondary.key_size
    if (primary.size[1] > secondary.size[1]):
        logging.warning('Normalizing layout size from secondary to primary')
        secondary.set_size(primary.size, surface_size)
    elif (primary.size[1] < secondary.size[1]):
        logging.warning('Normalizing layout size from primary to secondary')
        primary.set_size(secondary.size, surface_size)