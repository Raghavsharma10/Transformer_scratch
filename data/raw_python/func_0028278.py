def _normalize_items(
    ctx,
    items,
    str_to_node=False,
    node_to_str=False,
    allow_task=False,
):
    """
    Normalize given items.

    Do several things:
        - Ignore None.
        - Flatten list.
        - Unwrap wrapped item in `_ItemWrapper`.

    :param ctx: BuildContext object.

    :param items: Items list to normalize.

    :param str_to_node: Convert string to node.

    :param node_to_str: Convert node to absolute path.

    :param allow_task: Whether allow task item.

    :return: Normalized tuples list.

        Tuple format is:
        ::

            (
                normalized_item,        # Normalized item.
                wrapper_type,           # Original `_ItemWrapper` type.
            )
    """
    # Ensure given context object is BuildContext object
    _ensure_build_context(ctx)

    # Normalized tuples list
    norm_tuple_s = []

    # If given items list is empty
    if not items:
        # Return empty list
        return norm_tuple_s

    # If given items list is not empty.

    # For given items list's each item
    for item in items:
        # If the item is item wrapper
        if isinstance(item, _ItemWrapper):
            # Get wrapper type
            wrapper_type = item.type()

            # Get real item
            item = item.item()

        # If the item is not item wrapper
        else:
            # Set wrapper type be None
            wrapper_type = None

            # Use the item as real item
            item = item

        # If the real item is list
        if isinstance(item, list):
            # Use the real item as real items list
            real_item_s = item

        # If the real item is not list
        else:
            # Create real items list containing the real item
            real_item_s = [item]

        # For each real item
        for real_item in real_item_s:
            # If the real item is None
            if real_item is None:
                # Ignore None
                continue

            # If the real item is not None.

            # If the real item is string
            elif isinstance(real_item, str):
                # If need convert string to node
                if (wrapper_type is not None) or str_to_node:
                    # If the path string is absolute path
                    if os.path.isabs(real_item):
                        # Get error message
                        msg = (
                            'Error (7MWU9): Given path is not relative path:'
                            ' {0}.'
                        ).format(real_item)

                        # Raise error
                        raise ValueError(msg)

                    # If the path string is not absolute path.

                    # Create node as normalized item
                    norm_item = create_node(ctx, real_item)

                    # If need convert node to absolute path
                    if node_to_str:
                        # Convert the node to absolute path
                        norm_item = norm_item.abspath()

                # If not need convert string to node
                else:
                    # Use the string as normalized item
                    norm_item = real_item

                # Create normalized tuple
                norm_tuple = (norm_item, wrapper_type)

            # If the real item is not string.

            # If the real item is node
            elif isinstance(real_item, Node):
                # If need convert node to absolute path
                if node_to_str:
                    # Convert the node to absolute path
                    real_item = real_item.abspath()

                # Create normalized tuple
                norm_tuple = (real_item, wrapper_type)

            # If the real item is not node.

            # If the real item is task
            elif isinstance(real_item, Task):
                # If allow task item
                if allow_task:
                    # Create normalized tuple
                    norm_tuple = (real_item, wrapper_type)

                # If not allow task item
                else:
                    # Get error message
                    msg = 'Error (6PVMG): Item type is not valid: {0}.'.format(
                        real_item
                    )

                    # Raise error
                    raise ValueError(msg)

            # If the real item is not task.

            # If the real item is not None, string, node, or task
            else:
                # Get error message
                msg = 'Error (63KUG): Item type is not valid: {0}.'.format(
                    real_item
                )

                # Raise error
                raise ValueError(msg)

            # Add the normalized tuple to list
            norm_tuple_s.append(norm_tuple)

    # Return the normalized tuples list
    return norm_tuple_s