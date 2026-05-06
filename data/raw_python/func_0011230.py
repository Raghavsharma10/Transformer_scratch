def fixed_development_directory(templates, inventory, topics, user):
    """Return absolute path to development directory

    Arguments:
        templates (dict): templates.yaml
        inventory (dict): inventory.yaml
        context (dict): The be context, from context()
        topics (list): Arguments to `in`
        user (str): Current `be` user

    """

    echo("Fixed syntax has been deprecated, see positional syntax")

    project, item, task = topics[0].split("/")

    template = binding_from_item(inventory, item)
    pattern = pattern_from_template(templates, template)

    if find_positional_arguments(pattern):
        echo("\"%s\" uses a positional syntax" % project)
        echo("Try this:")
        echo("  be in %s" % " ".join([project, item, task]))
        sys.exit(USER_ERROR)

    keys = {
        "cwd": os.getcwd(),
        "project": project,
        "item": item.replace("\\", "/"),
        "user": user,
        "task": task,
        "type": task,  # deprecated
    }

    try:
        return pattern.format(**keys).replace("\\", "/")
    except KeyError as exc:
        echo("TEMPLATE ERROR: %s is not an available key\n" % exc)
        echo("Available keys")
        for key in keys:
            echo("\n- %s" % key)
        sys.exit(1)