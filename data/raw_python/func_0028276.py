def _mark_target(type, item):
    """
    Wrap given item as input or output target that should be added to task.

    Wrapper object will be handled specially in \
    :paramref:`create_cmd_task.parts`.

    :param type: Target type.

        Allowed values:
            - 'input'
            - 'output'

    :param item: Item to mark as input or output target.

        Allowed values:
            - Relative path relative to top directory.
            - Node object.
            - List of these.

    :return: Wrapper object.
    """
    # If given type is not valid
    if type not in ('input', 'output'):
        # Get error message
        msg = 'Error (7D74X): Type is not valid: {0}'.format(type)

        # Raise error
        raise ValueError(msg)

    # If given type is valid.

    # Store given item
    orig_item = item

    # If given path is list
    if isinstance(item, list):
        # Use it as items list
        item_s = item

    # If given path is not list
    else:
        # Create items list containing given path
        item_s = [item]

    # For the items list's each item
    for item in item_s:
        # If the item is string,
        # and the item is absolute path.
        if isinstance(item, str) and os.path.isabs(item):
            # Get error message
            msg = (
                'Error (5VWOZ): Given path is not relative path: {0}.'
            ).format(item)

            # Raise error
            raise ValueError(msg)

    # Wrap given item
    return _ItemWrapper(type=type, item=orig_item)