def pos_development_directory(templates,
                              inventory,
                              context,
                              topics,
                              user,
                              item):
    """Return absolute path to development directory

    Arguments:
        templates (dict): templates.yaml
        inventory (dict): inventory.yaml
        context (dict): The be context, from context()
        topics (list): Arguments to `in`
        user (str): Current `be` user
        item (str): Item from template-binding address

    """

    replacement_fields = replacement_fields_from_context(context)
    binding = binding_from_item(inventory, item)
    pattern = pattern_from_template(templates, binding)

    positional_arguments = find_positional_arguments(pattern)
    highest_argument = find_highest_position(positional_arguments)
    highest_available = len(topics) - 1
    if highest_available < highest_argument:
        echo("Template for \"%s\" requires at least %i arguments" % (
            item, highest_argument + 1))
        sys.exit(USER_ERROR)

    try:
        return pattern.format(*topics, **replacement_fields).replace("\\", "/")
    except KeyError as exc:
        echo("TEMPLATE ERROR: %s is not an available key\n" % exc)
        echo("Available tokens:")
        for key in replacement_fields:
            echo("\n- %s" % key)
        sys.exit(TEMPLATE_ERROR)